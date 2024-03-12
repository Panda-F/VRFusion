import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .data import cfg, mask_type
from .layers.box_utils import crop, sanitize_coordinates, center_size
from .layers.output_utils import display_lincomb, undo_image_transformation
from .utils import timer
from .utils.augmentations import FastBaseTransform, Resize
from .yolact import Yolact

COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))


class InstancePerception:
    def __init__(self, model_path):
        self.net = Yolact()
        self.net.load_weights(model_path)
        self.net.eval()
        self.net.cuda()

    def infer(self, img):
        h, w, _ = img.shape
        frame = torch.from_numpy(img).float().cuda()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        with torch.no_grad():
            dets_out = self.net(batch.cuda())
        dets_out = self.postprocess(dets_out, w, h)
        return dets_out

    def draw(self, img):
        h, w, _ = img.shape
        frame = torch.from_numpy(img).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        with torch.no_grad():
            dets_out = self.net(batch)
        result = self.prep_display(dets_out, frame, h, w)
        return result

    def postprocess(self, det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                    visualize_lincomb=False, crop_masks=True, score_threshold=0):
        """
        Postprocesses the output of Yolact on testing mode into a format that makes sense,
        accounting for all the possible configuration settings.

        Args:
            - det_output: The lost of dicts that Detect outputs.
            - w: The real with of the image.
            - h: The real height of the image.
            - batch_idx: If you have multiple images for this batch, the image's index in the batch.
            - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

        Returns 4 torch Tensors (in the following order):
            - classes [num_det]: The class idx for each detection.
            - scores  [num_det]: The confidence score for each detection.
            - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
            - masks   [num_det, h, w]: Full image masks for each detection.
        """

        dets = det_output[batch_idx]

        if dets is None:
            return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

        if score_threshold > 0:
            keep = dets['score'] > score_threshold

            for k in dets:
                if k != 'proto':
                    dets[k] = dets[k][keep]

            if dets['score'].size(0) == 0:
                return [torch.Tensor()] * 4

        # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
        b_w, b_h = (w, h)

        # Undo the padding introduced with preserve_aspect_ratio
        if cfg.preserve_aspect_ratio:
            r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)

            # Get rid of any detections whose centers are outside the image
            boxes = dets['box']
            boxes = center_size(boxes)
            s_w, s_h = (r_w / cfg.max_size, r_h / cfg.max_size)

            not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1  # not (a or b)
            for k in dets:
                if k != 'proto':
                    dets[k] = dets[k][not_outside]

            # A hack to scale the bboxes to the right size
            b_w, b_h = (cfg.max_size / r_w * w, cfg.max_size / r_h * h)

        # Actually extract everything from dets now
        classes = dets['class']
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']

        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            # At this points masks is only the coefficients
            proto_data = dets['proto']

            # Test flag, do not upvote
            # if cfg.mask_proto_debug:
            #     np.save('scripts/proto.npy', proto_data.cpu().numpy())

            if visualize_lincomb:
                display_lincomb(proto_data, masks)

            masks = torch.matmul(proto_data, masks.t())
            masks = cfg.mask_proto_mask_activation(masks)

            # Crop masks before upsampling because you know why
            if crop_masks:
                masks = crop(masks, boxes)

            # Permute into the correct output shape [num_dets, proto_h, proto_w]
            masks = masks.permute(2, 0, 1).contiguous()

            # Scale masks up to the full image
            if cfg.preserve_aspect_ratio:
                # Undo padding
                masks = masks[:, :int(r_h / cfg.max_size * proto_data.size(1)),
                        :int(r_w / cfg.max_size * proto_data.size(2))]

            masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

            # Binarize the masks
            masks.gt_(0.5)

        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
        boxes = boxes.long()

        if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
            # Upscale masks
            full_masks = torch.zeros(masks.size(0), h, w)

            for jdx in range(masks.size(0)):
                x1, y1, x2, y2 = boxes[jdx, :]

                mask_w = x2 - x1
                mask_h = y2 - y1

                # Just in case
                if mask_w * mask_h <= 0 or mask_w < 0:
                    continue

                mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
                mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
                mask = mask.gt(0.5).float()
                full_masks[jdx, y1:y2, x1:x2] = mask

            masks = full_masks

        return classes, scores, boxes, masks

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy)
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            t = self.postprocess(dets_out, w, h)
            # torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:5]
            classes, scores, boxes = [x[:5].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(5, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < 0:
                num_dets_to_consider = j
                break

        if num_dets_to_consider == 0:
            # No detections found so just output the original image
            return (img_gpu * 255).byte().cpu().numpy()

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if True and cfg.eval_mask_branch:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat(
                [(torch.Tensor(get_color(j)).float() / 255).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if True:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if True:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if True:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if True else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                cv2.LINE_AA)

        return img_numpy


if __name__ == "__main__":
    model_path = "../checkpoints/yolact_im700_54_800000.pth"
    instancePerception = InstancePerception(model_path)
    path = "../../static/materials/screenshot.png"
    img = cv2.imread(path)
    # mask = instancePerception.draw(img)
    # plt.figure("123")
    # plt.imshow(mask)
    # plt.show()
    classes, scores, boxes, masks = instancePerception.infer(img)
    for index, mask in enumerate(masks):
        print(mask)
        print(cfg.dataset.class_names[classes[index]])
        break
