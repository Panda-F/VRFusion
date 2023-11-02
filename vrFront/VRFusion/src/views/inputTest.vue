<template>
  <div>
    <div class="content">
      <el-text class="mx-1" size="large">原始图片</el-text><br>
      <el-upload
          class="upload-demo"
          drag
          action="http://127.0.0.1:80/save_img/"
          :show-file-list="false"
          :data="fileData1"
          :headers="headers"
          :on-success="handleAvatarSuccess1"
          :before-upload="beforeAvatarUpload"
      >
        <img v-if="imageUrl1" :src="imageUrl1" class="upload-image"/>
        <div v-else>
          <el-icon class="el-icon--upload">
            <upload-filled/>
          </el-icon>
          <div class="el-upload__text">
            Drop file here or <em>click to upload</em>
          </div>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            jpg/png files with a size less than 500kb
          </div>
        </template>

      </el-upload>
    </div>
    <div class="content">
      <el-text class="mx-1" size="large">待融合素材</el-text><br>
      <el-upload
          class="upload-demo"
          drag
          action="http://127.0.0.1:80/save_img/"
          :show-file-list="false"
          :data="fileData2"
          :headers="headers"
          :on-success="handleAvatarSuccess2"
          :before-upload="beforeAvatarUpload"
      >
        <img v-if="imageUrl2" :src="imageUrl2" class="upload-image"/>
        <div v-else>
          <el-icon class="el-icon--upload">
            <upload-filled/>
          </el-icon>
          <div class="el-upload__text">
            Drop file here or <em>click to upload</em>
          </div>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            jpg/png files with a size less than 500kb
          </div>
        </template>
      </el-upload>
    </div>
    <br>
    <img v-if="imageResult" :src="imageResult" class="upload-image"/>
    <br>
    <el-button v-on:click="fusion_image">虚实融合</el-button>

  </div>
</template>

<script>

import {fusionImage} from "../service/axiosInstance.js";

export default {
  name: "inputTest",
  data() {
    return {
      imageUrl1: "",
      fileData1: "",
      headers: "",
      imageUrl2: "",
      fileData2: "",
      imageResult:""
    }
  },
  methods: {
    // 图片上传成功的操作
    handleAvatarSuccess1(res) {
      if (res.code === 200) {
        console.log(res.static_url)
        this.imageUrl1 = res.static_url
      }
    },
    handleAvatarSuccess2(res) {
      if (res.code === 200) {
        console.log(res.static_url)
        this.imageUrl2 = res.static_url
      }
    },
    // 图片上传前的判断
    beforeAvatarUpload(file) {
      let imgType = ['jpg', 'jpeg', 'png']
      let judge = false // 后缀
      let type = file.name.split('.')[file.name.split('.').length - 1]
      for (let k = 0; k < imgType.length; k++) {
        if (imgType[k].toUpperCase() === type.toUpperCase()) {
          judge = true
          break
        }
      }
      // 验证图片格式
      if (!judge) {
        this.$message.error('图片格式只支持：JPG、JPEG、PNG')
        return false
      }
      const isLt1M = file.size / 1024 / 1024
      if (isLt1M > 1) {
        this.$message.error('上传头像图片大小不能超过1MB')
        return false
      }
      return true
    },
    // 融合图片
    fusion_image(){
      let param = {background:this.imageUrl1, front:this.imageUrl2}
      fusionImage(param).then(res=>{
        this.imageResult = res.data['static_url']
      })
    }
  },
  created() {

  },
  mounted() {

  }
}
</script>

<style scoped>
.upload-image {
  width: 300px;
  height: 300px;
}
.content{
  display: inline-block;
}
.upload-demo {
  display: inline-block;
  padding: 20px;
}
</style>