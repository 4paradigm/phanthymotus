#ifndef MOTUS_QUEST_CAPTURE_MBEDTLS_USER_CONFIG_H_
#define MOTUS_QUEST_CAPTURE_MBEDTLS_USER_CONFIG_H_

// libdatachannel's WebRTC transport negotiates DTLS-SRTP (RFC 5764).
// Mbed TLS 3.6 keeps this optional in its default configuration, so the APK
// enables it explicitly through the supported user-config hook.
#define MBEDTLS_SSL_DTLS_SRTP

#endif  // MOTUS_QUEST_CAPTURE_MBEDTLS_USER_CONFIG_H_
