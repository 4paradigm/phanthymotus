import java.security.MessageDigest
import java.util.Properties

plugins {
    id("com.android.application")
}

val ndkNoticeAssets = layout.buildDirectory.dir("generated/ndkNoticeAssets")
val pinnedNdkRevision = "27.0.12077973"

android {
    namespace = "com.phanthymotus.questcapture"
    compileSdk = 35
    buildToolsVersion = "35.0.1"
    ndkVersion = pinnedNdkRevision

    defaultConfig {
        applicationId = "com.phanthymotus.questcapture"
        minSdk = 29
        targetSdk = 35
        versionCode = 22
        versionName = "0.3.18-onboarding1"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++20", "-fexceptions", "-frtti")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                )
                targets += "motus_openxr_capture"
            }
        }
    }

    flavorDimensions += "headset"
    productFlavors {
        create("meta") {
            dimension = "headset"
            applicationId = "com.phanthymotus.questcapture"
            externalNativeBuild {
                cmake {
                    arguments += "-DMOTUS_CAPTURE_HEADSET=meta"
                }
            }
        }
        create("pico") {
            dimension = "headset"
            applicationId = "com.phanthymotus.picocapture"
            externalNativeBuild {
                cmake {
                    arguments += "-DMOTUS_CAPTURE_HEADSET=pico"
                }
            }
        }
    }

    buildFeatures {
        prefab = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    signingConfigs {
        create("distribution") {
            val path = System.getenv("MOTUS_APK_KEYSTORE")
            if (!path.isNullOrEmpty()) {
                storeFile = file(path)
                storePassword = System.getenv("MOTUS_APK_STORE_PASSWORD")
                keyAlias = System.getenv("MOTUS_APK_KEY_ALIAS")
                keyPassword = System.getenv("MOTUS_APK_KEY_PASSWORD")
            }
        }
    }
    buildTypes {
        debug {
            isJniDebuggable = true
        }
        release {
            signingConfig = signingConfigs.getByName("distribution")
            isMinifyEnabled = false
        }
    }
    packaging {
        jniLibs.useLegacyPackaging = true
    }
    sourceSets.getByName("main").assets.srcDir(ndkNoticeAssets)
}

dependencies {
    implementation("org.khronos.openxr:openxr_loader_for_android:1.1.60")
}

val prepareNdkNotices by tasks.registering {
    val ndk = androidComponents.sdkComponents.sdkDirectory.map { it.dir("ndk/$pinnedNdkRevision") }
    val sources = mapOf(
        "NOTICE" to "4d5224d1c0b54ffa88b0dc0088191638fdbb7227835095dfb98f7f66b416a323",
        "NOTICE.toolchain" to "cbe3237be53c0a819f8df6aac5358fdee848eee3b6571b4e2ca20767ebd7465e",
    )
    inputs.files(ndk.map { root -> sources.keys.map { root.file(it) } })
    inputs.file(ndk.map { it.file("source.properties") })
    outputs.dir(ndkNoticeAssets)
    doLast {
        val root = ndk.get().asFile
        val properties = Properties().apply {
            root.resolve("source.properties").inputStream().use { load(it) }
        }
        check(properties.getProperty("Pkg.Revision") == pinnedNdkRevision) {
            "NDK notice source revision mismatch"
        }
        // Verify every input before materializing complete, unmodified texts.
        val verified = sources.mapValues { (name, expected) ->
            root.resolve(name).readBytes().also { bytes ->
                val actual = MessageDigest.getInstance("SHA-256").digest(bytes)
                    .joinToString("") { "%02x".format(it) }
                check(actual == expected) { "NDK notice checksum mismatch: $name" }
            }
        }
        val destination = ndkNoticeAssets.get().asFile.resolve("licenses")
        destination.mkdirs()
        verified.forEach { (name, bytes) -> destination.resolve("Android-NDK-$name").writeBytes(bytes) }
    }
}

val verifyOpenXrArtifact by tasks.registering {
    val artifact = configurations.detachedConfiguration(
        dependencies.create("org.khronos.openxr:openxr_loader_for_android:1.1.60"),
    )
    inputs.files(artifact)
    doLast {
        val file = artifact.singleFile
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(file.readBytes())
            .joinToString("") { "%02x".format(it) }
        check(digest == "9a21ecea6b308d3a7fcf261412bec4cb1ae9148ba053d6b24da468fef96029c7") {
            "OpenXR Android loader checksum mismatch: $digest"
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn(verifyOpenXrArtifact, prepareNdkNotices)
}
