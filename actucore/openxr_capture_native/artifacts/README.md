此目录只保存本地构建验证产物，不是普通镜像构建的额外前置步骤。`scripts/stage_apk.py` 校验实际 APK 签名、包名、对齐并生成本地 SHA256 清单。APK 和本地清单不提交 Git。

普通 ActuCore Docker 构建自动读取仓库中的 `package-manifest.json`，从固定 HTTPS URL 下载正式 APK 并校验大小及 SHA256；下载器不接受运行时用户 URL、不回退 debug 或未校验制品。制品更新流程是构建与验证 release APK、发布到不可变地址、再更新固定清单；私钥始终留在外部私有目录。
