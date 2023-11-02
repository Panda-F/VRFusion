<template>
  <div class="imageUpload">
    <el-upload
        class="avatar-uploader"
        action=""
        list-type="picture"
        accept=".jpg, .png"
        :limit="1"
        :auto-upload="false"
        :file-list="fileList"
        :on-change="getFile"
        :on-preview="handlePictureCardPreview"
        :on-remove="handleUploadRemove"
    >
      <img v-if="imageUrl" :src="imageUrl" class="avatar">
      <el-icon v-else class="avatar-uploader-icon">
        <Plus/>
      </el-icon>
    </el-upload>
  </div>


</template>
<script>
import {Plus} from '@element-plus/icons-vue'
import API from "../service/axiosInstance.js";

export default {
  name: "imageUpload",
  components: [Plus],
  data() {
    return {
      fileList: [],
      imageUrl: '',
      ruleForm: ''
    }
  },
  methods: {
    getFile(file, fileList) {
      this.getBase64(file.raw).then(res => {
        const params = res
        this.proofImage = params
        this.sendData()

      })

    },
    getBase64(file) {
      return new Promise(function (resolve, reject) {
        const reader = new FileReader()
        let imgResult = ''
        reader.readAsDataURL(file)
        reader.onload = function () {
          imgResult = reader.result
        }
        reader.onerror = function (error) {
          reject(error)
        }
        reader.onloadend = function () {
          resolve(imgResult)
        }
      })
    },
    handleUploadRemove(file, fileList) {
      this.proofImage = ''
      this.ruleForm.message_img = ''
    },
    handlePictureCardPreview(file) {
      console.log(this.proofImage)
    },
    sendData() {
      API({
            method: "post",
            // url: 'http://127.0.0.1:8000/imageSynthesis',
            url: 'http://127.0.0.1:8000/test',
            headers: {"Content-Type": "application/json"},
            data: {
              "input_image": this.proofImage,
            }
          }
      ).then((value) => {
        this.instances = value.data["instances"]
        this.$emit('sendInstance',this.instances)
      })
    }
  },


}
</script>
<style scoped>
.avatar-uploader .avatar {
  width: 178px;
  height: 178px;
  display: block;
}
</style>

<style>
.avatar-uploader .el-upload {
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: var(--el-transition-duration-fast);
}

.avatar-uploader .el-upload:hover {
  border-color: var(--el-color-primary);
}

.el-icon.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  text-align: center;
}
</style>