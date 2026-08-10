import java.security.MessageDigest

plugins {
    id("com.android.application")
}

android {
    namespace = "com.phanthymotus.questcapture"
    compileSdk = 35
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.phanthymotus.questcapture"
        minSdk = 29
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"

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
                targets += "motus_quest_capture"
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
    buildTypes {
        debug {
            isJniDebuggable = true
        }
        release {
            isMinifyEnabled = false
        }
    }
    packaging {
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    implementation("org.khronos.openxr:openxr_loader_for_android:1.1.60")
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
    dependsOn(verifyOpenXrArtifact)
}
