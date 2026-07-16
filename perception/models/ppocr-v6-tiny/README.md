# PP-OCRv6 Tiny 离线模型

该目录本地应包含以下文件，但模型文件已由 `.gitignore` 排除：

```text
det.onnx
rec.onnx
inference.yml
```

- `det.onnx`：PP-OCRv6 Tiny 文本检测模型。
- `rec.onnx`：PP-OCRv6 Tiny 多语言文本识别模型。
- `inference.yml`：识别模型配置及字符字典。

上传到模型服务器时保持以上文件名。正式镜像只通过
`plugins.ocr.model_base_url` 从内部 HTTP 服务下载，不使用公共模型源：

```yaml
model_base_url: "http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny"
```

模型总大小必须小于 15 MiB。

构建镜像时指定地址：

```bash
docker build \
  --build-arg OCR_MODEL_BASE_URL=http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny \
  -f perception/Dockerfile.jetson \
  .
```

运行时也可以覆盖镜像内的默认值：

```bash
docker run \
  -e OCR_MODEL_BASE_URL=http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny \
  <image>
```
