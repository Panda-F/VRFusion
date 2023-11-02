<template>
  <div id="home">
    <h1>图像处理平台</h1>
    <imageUpload @sendInstance="getInstance"></imageUpload>
    <el-button v-on:click="handelInstance">处理图层</el-button>
    <div id="map"></div>
    <div ref="overlay" class="overlay">
      <div class="overlay-content"></div>
    </div>
  </div>

</template>
<script>
import {Map, View, Feature} from "ol"
import {Image, Vector as VectorLayer} from "ol/layer.js"
import {Vector as VectorSource} from "ol/source"
import Static from 'ol/source/ImageStatic'
import ImageLayer from 'ol/layer/Image'
import {Stroke, Style, Text} from 'ol/style';
import {transform, fromLonLat, Projection} from 'ol/proj'
import {getCenter} from "ol/extent.js";
import {LineString, Point, Polygon} from "ol/geom";
import imageUpload from "../components/imageUpload.vue";

export default {
  components:{imageUpload},
  data() {
    return {
      map: null,
      extent: null,

    };
  },
  methods: {
    initImage() {
      this.extent = [0, 0, 640, 480]  // 图片范围
      this.projection = new Projection({  // 创建投影
        code: 'xkcd-image',
        units: 'pixels',   // 像素单位
        extent: this.extent
      })

      this.map = new Map({
        layers: [
          new ImageLayer({
            title: 'baseMap',
            source: new Static({
              url: 'http://localhost:8080/src/static/images/dog.jpg',
              projection: this.projection,
              imageExtent: this.extent
            })
          })
        ],
        target: 'map',   // 绑定的地图显示元素
        view: new View({
          projection: this.projection,
          center: getCenter(this.extent),
          zoom: 2,
          maxZoom: 4
        })
      })
    },
    addMask(mask,label,id) {
      // 创建vector对象
      this.vectorLayer = new VectorLayer({
        source: new VectorSource(),
        opacity: 0.6,
        name:label+id
      })
      // 创建feature对象
      this.feature = new Feature({
        geometry: new Polygon([
          mask
        ]),
      })
      this.feature.setStyle(new Style({
        text: new Text({
          font:'13px Microsoft YaHei',
          text: label+id,
          textAlign:"left",
          textBaseline:"top",
          scale:1
        }),
        stroke: new Stroke({
          color: '#ffcc33',
          width: 1,
        }),
      }))
      this.vectorLayer.getSource().addFeature(
          //添加点图层
          // new Feature({
          //   geometry: new Point(getCenter(this.extent)),
          // })
          //添加线图层
          // new Feature({
          //   geometry: new LineString([
          //     getCenter(this.extent),
          //     [300,200],
          //   ]),
          // })
          //添加面图层
          this.feature
      );
      this.map.addLayer(this.vectorLayer);

    },
    getInstance(val){
      this.instances=val
    },
    handelInstance(){
      console.log("处理图层")
      for(let i in this.instances){
        console.log(this.instances[i])
        if(this.instances[i]["score"] > 0.08){
          this.addMask(this.instances[i]["mask"],this.instances[i]["class"],i)
        }
      }
    }
  },
  mounted() {
    this.initImage()
    // this.addMask()
  }
  ,
}
;

</script>

<style scoped>
#map {
  width: 640px;
  height: 480px;
  border: 2px solid black;
  border-radius: 3px;
  box-shadow: 2px 2px 2px black;
  margin-top: 100px;
}
</style>
