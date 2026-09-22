package com.phanthymotus.capture;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.util.Base64;
import android.widget.*;
import org.json.JSONObject;
import javax.net.ssl.*;
import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Normal launcher. Untrusted TLS is confined to enrollment; capture pins the approved CA. */
public final class ConnectionActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private volatile boolean cancelled;
    private volatile boolean confirmed;
    private int discoveryEpoch;
    private boolean resolving;
    private final java.util.ArrayDeque<NsdServiceInfo> resolveQueue = new java.util.ArrayDeque<>();
    private TextView status;
    private EditText address;
    private LinearLayout devices;
    private Button confirm;
    private NsdManager nsd;
    private NsdManager.DiscoveryListener discovery;
    private WifiManager.MulticastLock multicast;

    private SharedPreferences credentials() { return getSharedPreferences("motus_capture", MODE_PRIVATE); }
    private void message(String value) { runOnUiThread(() -> status.setText(value)); }
    private Button button(LinearLayout parent, String label, Runnable action) {
        Button button = new Button(this); button.setText(label);
        button.setOnClickListener(v -> action.run()); parent.addView(button); return button;
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this); layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 24, 32, 24); scroll.addView(layout); setContentView(scroll);
        TextView title = new TextView(this); title.setText("PhanthyMotus · 连接机器人"); title.setTextSize(24); layout.addView(title);
        TextView legend = new TextView(this);
        legend.setText("透视调试视图：绿=实测手臂，橙=期望手臂姿态（IK），粉=手柄映射目标。\n模型显示在眼前，不与现实机器人重合；Shadow不驱动机器人；天轶暂停时暗橙线保留最后有效姿态，过期实测线隐藏。");
        legend.setTextSize(16); layout.addView(legend);
        status = new TextView(this); status.setTextSize(18); layout.addView(status);
        devices = new LinearLayout(this); devices.setOrientation(LinearLayout.VERTICAL); layout.addView(devices);
        button(layout, "刷新局域网机器人", this::discover);
        address = new EditText(this); address.setSingleLine(true); address.setHint("备用地址，例如 robot.local:15741"); layout.addView(address);
        button(layout, "申请配对", () -> begin(address.getText().toString().trim()));
        confirm = button(layout, "指纹一致，在头显确认", () -> { confirmed = true; confirm.setEnabled(false); });
        confirm.setEnabled(false);
        button(layout, "连接已配对机器人", () -> launch(null));
        button(layout, "取消 / 断开", () -> { cancelled = true; confirmed = false; confirm.setEnabled(false); message("已取消连接；运动不会自动恢复"); });
        button(layout, "忘记设备", () -> new AlertDialog.Builder(this).setMessage("忘记本机凭据？机器人卡片内也需撤销旧配对，才能重新申请。")
            .setPositiveButton("忘记", (d,w) -> { cancelled = true; credentials().edit().clear().commit(); message("已忘记设备"); })
            .setNegativeButton("取消", null).show());
        discover();
        // A remembered identity must not hide connection management or enter XR.
        // In particular, a headset moved to another robot must remain usable offline.
        if (!credentials().getString("capture_credential", "").isEmpty())
            message("已有配对记录。请确认机器人后点击连接；更换机器人请先忘记设备。");
        savePreviousAnr();
    }

    private void savePreviousAnr() {
        if (android.os.Build.VERSION.SDK_INT < 30 ||
                (getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0) return;
        worker.execute(() -> {
            try {
                android.app.ActivityManager manager = (android.app.ActivityManager)getSystemService(ACTIVITY_SERVICE);
                for (android.app.ApplicationExitInfo exit : manager.getHistoricalProcessExitReasons(getPackageName(), 0, 4)) {
                    if (exit.getReason() != android.app.ApplicationExitInfo.REASON_ANR) continue;
                    try (InputStream input = exit.getTraceInputStream()) {
                        if (input == null) continue;
                        try (OutputStream output = openFileOutput("previous-anr.txt", MODE_PRIVATE)) {
                            byte[] buffer = new byte[8192]; int remaining = 2 * 1024 * 1024;
                            while (remaining > 0) {
                                int n = input.read(buffer, 0, Math.min(buffer.length, remaining));
                                if (n < 0) break;
                                output.write(buffer, 0, n); remaining -= n;
                            }
                        }
                    }
                    break;
                }
            } catch (Exception ignored) { /* Debug evidence must never block the launcher. */ }
        });
    }

    private void stopDiscovery() {
        discoveryEpoch++; resolving=false; resolveQueue.clear();
        if (nsd != null && discovery != null) {
            try { nsd.stopServiceDiscovery(discovery); } catch (IllegalArgumentException ignored) { }
            discovery = null;
        }
        if (multicast != null && multicast.isHeld()) multicast.release();
    }

    private void discover() {
        stopDiscovery(); devices.removeAllViews();
        message("正在发现机器人；未发现时检查同一局域网，或输入备用地址。先在卡片内打开允许配对。");
        nsd = (NsdManager)getSystemService(NSD_SERVICE);
        WifiManager wifi = (WifiManager)getApplicationContext().getSystemService(WIFI_SERVICE);
        multicast = wifi.createMulticastLock("motus-discovery"); multicast.setReferenceCounted(false); multicast.acquire();
        discovery = new NsdManager.DiscoveryListener() {
            public void onDiscoveryStarted(String type) { }
            public void onDiscoveryStopped(String type) { }
            public void onStartDiscoveryFailed(String type, int error) { message("自动发现不可用，请刷新或输入地址（"+error+"）"); }
            public void onStopDiscoveryFailed(String type, int error) { }
            public void onServiceLost(NsdServiceInfo service) {
                runOnUiThread(() -> { for (int i=devices.getChildCount()-1;i>=0;i--) if (service.getServiceName().equals(devices.getChildAt(i).getTag())) devices.removeViewAt(i); });
            }
            public void onServiceFound(NsdServiceInfo service) {
                runOnUiThread(() -> {
                    if (discovery != this || resolveQueue.size() >= 32) return;
                    resolveQueue.add(service); resolveNext();
                });
            }
        };
        nsd.discoverServices("_motus-teleop._tcp.", NsdManager.PROTOCOL_DNS_SD, discovery);
    }

    private void resolveNext() {
        if (resolving || resolveQueue.isEmpty() || discovery == null) return;
        resolving=true;
        final int epoch=discoveryEpoch;
        nsd.resolveService(resolveQueue.remove(), new NsdManager.ResolveListener() {
            public void onResolveFailed(NsdServiceInfo item,int code) {
                runOnUiThread(() -> { if(epoch!=discoveryEpoch)return; resolving=false; resolveNext(); });
            }
            public void onServiceResolved(NsdServiceInfo item) {
                runOnUiThread(() -> {
                    if(epoch!=discoveryEpoch)return;
                    resolving=false;
                    if(item.getHost()!=null && item.getPort()>0 && devices.getChildCount()<32) {
                        String host=item.getHost().getHostAddress();
                        final String endpoint=(host.contains(":")?"["+host+"]":host)+":"+item.getPort();
                        byte[] name=item.getAttributes().get("name");
                        String label=name==null?item.getServiceName():new String(name,StandardCharsets.UTF_8);
                        for(int i=devices.getChildCount()-1;i>=0;i--) if(item.getServiceName().equals(devices.getChildAt(i).getTag())) devices.removeViewAt(i);
                        Button b=button(devices,label+" · "+endpoint,()->{address.setText(endpoint);begin(endpoint);});
                        b.setTag(item.getServiceName());
                    }
                    resolveNext();
                });
            }
        });
    }

    private static String hex(byte[] bytes) {
        StringBuilder text = new StringBuilder(); for (byte b: bytes) text.append(String.format("%02X", b & 255)); return text.toString();
    }
    private static byte[] digest(byte[] bytes) throws Exception { return MessageDigest.getInstance("SHA-256").digest(bytes); }

    private static final class PairChannel {
        final String origin;
        X509Certificate certificate;
        final SSLSocketFactory sockets;
        PairChannel(String endpoint) throws Exception {
            URL parsed = new URL("https://"+endpoint);
            if (parsed.getHost().isEmpty() || parsed.getUserInfo()!=null || parsed.getQuery()!=null || parsed.getRef()!=null || !parsed.getPath().isEmpty() || parsed.getPort()<1)
                throw new IOException("请输入主机名或 IP 和端口");
            origin = parsed.toString();
            SSLContext tls = SSLContext.getInstance("TLS");
            tls.init(null, new TrustManager[]{new X509TrustManager() {
                public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[0]; }
                public void checkClientTrusted(X509Certificate[] chain,String type) throws java.security.cert.CertificateException { throw new java.security.cert.CertificateException(); }
                public void checkServerTrusted(X509Certificate[] chain,String type) throws java.security.cert.CertificateException {
                    if (chain.length==0) throw new java.security.cert.CertificateException();
                    chain[0].checkValidity();
                    if (certificate == null) certificate = chain[0];
                    else if (!Arrays.equals(certificate.getEncoded(),chain[0].getEncoded())) throw new java.security.cert.CertificateException("配对过程中证书变化");
                }
            }}, new SecureRandom());
            sockets = tls.getSocketFactory();
        }
        JSONObject post(String operation, JSONObject body) throws Exception {
            if (!operation.equals("request") && !operation.equals("poll")) throw new IOException();
            HttpsURLConnection c = (HttpsURLConnection)new URL(origin+"/pairing/"+operation).openConnection();
            c.setSSLSocketFactory(sockets);
            // Enrollment only: both displays bind the actual TLS leaf certificate before trusting it.
            c.setHostnameVerifier((host,session) -> true);
            c.setInstanceFollowRedirects(false); c.setConnectTimeout(5000); c.setReadTimeout(5000);
            c.setRequestMethod("POST"); c.setRequestProperty("Content-Type","application/json"); c.setDoOutput(true);
            byte[] payload=body.toString().getBytes(StandardCharsets.UTF_8); c.setFixedLengthStreamingMode(payload.length);
            try {
                try (OutputStream out=c.getOutputStream()) { out.write(payload); }
                int code=c.getResponseCode();
                InputStream input=code==200?c.getInputStream():c.getErrorStream();
                if (input==null) throw new IOException("配对服务不可用（"+code+"）");
                ByteArrayOutputStream bytes=new ByteArrayOutputStream();
                try (InputStream in=input) { byte[] block=new byte[4096]; int n;
                    while ((n=in.read(block))!=-1) { bytes.write(block,0,n); if(bytes.size()>65536) throw new IOException("响应过大"); }
                }
                JSONObject result=new JSONObject(bytes.toString("UTF-8"));
                if (code!=200) throw new IOException(result.optString("error","配对失败"));
                return result;
            } finally { c.disconnect(); }
        }
    }

    private volatile boolean pairing;
    private void begin(String endpoint) {
        if (pairing) { message("已有配对申请，请先完成或取消后重试"); return; }
        if (!credentials().getString("capture_credential", "").isEmpty()) { message("已有配对，请先忘记设备并在卡片内撤销旧配对"); return; }
        cancelled=false; confirmed=false; pairing=true; confirm.setEnabled(false);
        worker.execute(() -> {
            try {
                PairChannel channel = new PairChannel(endpoint);
                KeyPairGenerator generator=KeyPairGenerator.getInstance("EC"); generator.initialize(256);
                byte[] key=generator.generateKeyPair().getPublic().getEncoded();
                byte[] nonce=new byte[32]; new SecureRandom().nextBytes(nonce);
                JSONObject request=new JSONObject().put("device_name",android.os.Build.MODEL)
                    .put("public_key",Base64.encodeToString(key,Base64.NO_WRAP)).put("nonce",Base64.encodeToString(nonce,Base64.NO_WRAP));
                JSONObject pending=channel.post("request",request);
                ByteArrayOutputStream transcript=new ByteArrayOutputStream();
                transcript.write("motus-enrollment-v1\0".getBytes(StandardCharsets.UTF_8));
                transcript.write(digest(channel.certificate.getEncoded())); transcript.write(digest(key)); transcript.write(nonce);
                byte[] serverNonce=Base64.decode(pending.getString("server_nonce"),Base64.NO_WRAP);
                if(serverNonce.length!=32) throw new IOException("配对响应无效");
                transcript.write(serverNonce);
                String fingerprint=hex(digest(transcript.toByteArray())).substring(0,32);
                if (!fingerprint.equals(pending.getString("fingerprint"))) throw new IOException("服务器身份校验失败");
                message("请与 Canvas 卡片逐组核对指纹：\n"+fingerprint.replaceAll("(.{8})(?!$)","$1 ")+"\n两端确认后才能配对；不一致请取消。");
                runOnUiThread(() -> confirm.setEnabled(!cancelled));
                long deadline=android.os.SystemClock.elapsedRealtime()+120000;
                while (!cancelled && android.os.SystemClock.elapsedRealtime()<deadline) {
                    JSONObject poll=new JSONObject().put("request_id",pending.getString("request_id"))
                        .put("ticket",pending.getString("ticket")).put("confirm",confirmed).put("fingerprint",fingerprint);
                    JSONObject result=channel.post("poll",poll);
                    if (result.optString("state").equals("approved")) {
                        byte[] pem=Base64.decode(result.getString("ca_certificate_base64"),Base64.DEFAULT);
                        X509Certificate cert=(X509Certificate)java.security.cert.CertificateFactory.getInstance("X.509").generateCertificate(new ByteArrayInputStream(pem));
                        if (!Arrays.equals(cert.getEncoded(),channel.certificate.getEncoded())) throw new IOException("批准的证书不一致");
                        if (!cancelled) runOnUiThread(() -> { if(!cancelled) launch(result); });
                        return;
                    }
                    Thread.sleep(1000);
                }
                message(cancelled ? "已取消配对" : "配对已过期，请重新打开卡片配对窗口");
            } catch (Exception e) { message("连接失败："+e.getMessage()); }
            finally { pairing=false; runOnUiThread(() -> confirm.setEnabled(false)); }
        });
    }

    private void launch(JSONObject pairingResult) {
        if (pairingResult==null && credentials().getString("capture_credential", "").isEmpty()) { message("请先配对机器人"); return; }
        Intent intent = new Intent(this, android.app.NativeActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        if (pairingResult!=null) {
            intent.putExtra("driver_capture_wss_url",pairingResult.optString("wss_url"));
            intent.putExtra("pairing_id",pairingResult.optString("pairing_id"));
            intent.putExtra("pairing_code",pairingResult.optString("pairing_code"));
            intent.putExtra("ca_certificate_base64",pairingResult.optString("ca_certificate_base64"));
        }
        message("正在进入透视采集。连接不会自动使能运动；返回此页可管理连接。");
        startActivityForResult(intent, 42);
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request,result,data);
        if(request==42) message("采集连接已结束。若连接失败，请检查网络；身份变化或配对被撤销时需重新配对。可点击连接重试。");
    }
    @Override public void onPause() { cancelled=true; stopDiscovery(); super.onPause(); }
    @Override public void onDestroy() { cancelled=true; stopDiscovery(); worker.shutdownNow(); super.onDestroy(); }
}
