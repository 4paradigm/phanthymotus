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
    private TextView discoveryStatus;
    private Button invitationConnect;
    private Button reconnect;
    private EditText address;
    private LinearLayout devices;
    private Button confirm;
    private JSONObject invitation;
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
        TextView help = new TextView(this);
        help.setText("首次连接：\n1. 头显和机器人连接同一局域网。\n2. 在电脑端遥操卡片点击“允许新设备配对”。\n3. 在下方选择机器人，核对两端配对码并确认。\n连接资料会自动保存，下次打开无需重新配置。");
        help.setTextSize(20); layout.addView(help);
        discoveryStatus = new TextView(this); discoveryStatus.setTextSize(20); layout.addView(discoveryStatus);
        devices = new LinearLayout(this); devices.setOrientation(LinearLayout.VERTICAL); layout.addView(devices);
        button(layout, "重新查找机器人", this::discover);
        status = new TextView(this); status.setTextSize(22); layout.addView(status);
        invitationConnect = button(layout, "连接邀请中的机器人", this::redeemInvitation);
        invitationConnect.setVisibility(android.view.View.GONE);
        confirm = button(layout, "两端配对码一致，确认连接", () -> { confirmed = true; confirm.setEnabled(false); message("已确认，等待电脑端批准连接…"); });
        confirm.setEnabled(false); confirm.setVisibility(android.view.View.GONE);
        reconnect = button(layout, "重新连接已配对机器人", () -> launch(null));
        reconnect.setVisibility(credentials().getString("capture_credential", "").isEmpty() ? android.view.View.GONE : android.view.View.VISIBLE);
        button(layout, "取消连接", () -> { cancelled = true; confirmed = false; confirm.setEnabled(false); message("已取消；需要连接时请重新选择机器人。"); });
        LinearLayout advanced = new LinearLayout(this); advanced.setOrientation(LinearLayout.VERTICAL);
        advanced.setVisibility(android.view.View.GONE);
        button(layout, "连接帮助 / 更换机器人", () -> advanced.setVisibility(advanced.getVisibility()==android.view.View.VISIBLE ? android.view.View.GONE : android.view.View.VISIBLE));
        layout.addView(advanced);
        TextView networkHelp = new TextView(this); networkHelp.setTextSize(18);
        networkHelp.setText("没有发现机器人：检查访客 Wi-Fi 或网络隔离。请维护者提供备用地址，或从安装页打开连接邀请。更换机器人时，请先在电脑端撤销旧配对，再忘记本机记录。");
        advanced.addView(networkHelp);
        address = new EditText(this); address.setSingleLine(true); address.setHint("维护者提供的备用地址"); advanced.addView(address);
        button(advanced, "通过备用地址连接", () -> begin(address.getText().toString().trim()));
        button(advanced, "忘记已配对机器人", () -> new AlertDialog.Builder(this).setMessage("忘记本机凭据？机器人卡片内也需撤销旧配对，才能重新申请。")
            .setPositiveButton("忘记", (d,w) -> { cancelled = true; credentials().edit().clear().commit(); reconnect.setVisibility(android.view.View.GONE); message("已忘记设备，请重新选择机器人。"); })
            .setNegativeButton("取消", null).show());
        discover();
        // Cold launch reconnects only to the persisted, certificate-pinned identity.
        // Returning from XR does not auto-launch again; connection management stays reachable.
        if (getIntent().getData() != null) importInvitation(getIntent());
        else if (state == null && !credentials().getString("capture_credential", "").isEmpty())
            launch(null);
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
        discoveryStatus.setText("正在查找同一局域网的机器人…");
        final int searchEpoch = discoveryEpoch;
        discoveryStatus.postDelayed(() -> {
            if (searchEpoch == discoveryEpoch && devices.getChildCount() == 0)
                discoveryStatus.setText("还没有找到机器人。请检查同一 Wi-Fi，点击重新查找；仍未找到可展开连接帮助。");
        }, 8000);
        nsd = (NsdManager)getSystemService(NSD_SERVICE);
        WifiManager wifi = (WifiManager)getApplicationContext().getSystemService(WIFI_SERVICE);
        multicast = wifi.createMulticastLock("motus-discovery"); multicast.setReferenceCounted(false); multicast.acquire();
        discovery = new NsdManager.DiscoveryListener() {
            public void onDiscoveryStarted(String type) { }
            public void onDiscoveryStopped(String type) { }
            public void onStartDiscoveryFailed(String type, int error) { runOnUiThread(() -> { if (discovery == this) discoveryStatus.setText("暂时无法查找机器人，请点击重新查找或展开连接帮助。"); }); }
            public void onStopDiscoveryFailed(String type, int error) { }
            public void onServiceLost(NsdServiceInfo service) {
                runOnUiThread(() -> { if (discovery != this) return; for (int i=devices.getChildCount()-1;i>=0;i--) if (service.getServiceName().equals(devices.getChildAt(i).getTag())) devices.removeViewAt(i); });
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
                        Button b=button(devices,"连接 · "+label,()->{address.setText(endpoint);begin(endpoint);});
                        b.setTextSize(22); b.setContentDescription("连接机器人 "+label+"，地址 "+endpoint);
                        discoveryStatus.setText("请选择要连接的机器人；请先在电脑端允许新设备配对。");
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
        PairChannel(String endpoint) throws Exception { this(endpoint, null); }
        PairChannel(String endpoint, String expectedCertificate) throws Exception {
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
                    if (expectedCertificate != null) {
                        try {
                            if (!hex(digest(chain[0].getEncoded())).equalsIgnoreCase(expectedCertificate))
                                throw new java.security.cert.CertificateException("机器人证书与邀请不一致");
                        } catch (java.security.cert.CertificateException e) { throw e; }
                        catch (Exception e) { throw new java.security.cert.CertificateException(e); }
                    }
                    if (certificate == null) certificate = chain[0];
                    else if (!Arrays.equals(certificate.getEncoded(),chain[0].getEncoded())) throw new java.security.cert.CertificateException("配对过程中证书变化");
                }
            }}, new SecureRandom());
            sockets = tls.getSocketFactory();
        }
        JSONObject post(String operation, JSONObject body) throws Exception {
            if (!operation.equals("request") && !operation.equals("poll") && !operation.equals("invite")) throw new IOException();
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
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        importInvitation(intent);
    }

    private String pairedIdentity() throws Exception {
        String pem = credentials().getString("ca_certificate_pem", "");
        if (pem.isEmpty()) return "";
        X509Certificate cert = (X509Certificate)java.security.cert.CertificateFactory.getInstance("X.509")
            .generateCertificate(new ByteArrayInputStream(pem.getBytes(StandardCharsets.UTF_8)));
        return hex(digest(cert.getEncoded())).toLowerCase(java.util.Locale.ROOT);
    }

    private void importInvitation(Intent intent) {
        if (pairing) { message("配对申请进行中，请先取消后重新打开邀请"); return; }
        invitation = null;
        invitationConnect.setVisibility(android.view.View.GONE);
        try {
            invitation = ConnectionInvitation.parse(intent.getDataString());
            address.setText(invitation.getString("endpoint"));
            invitationConnect.setVisibility(android.view.View.VISIBLE);
            message("已导入机器人 " + invitation.getString("device_id").substring(0,12)
                + " 的一次性邀请。点击连接邀请中的机器人；连接不会使能运动。");
        } catch (Exception e) { message("连接邀请无效，请从当前机器人 Canvas 重新打开"); }
        // Never retain the bearer token in the Activity's reusable launch Intent.
        intent.setData(null);
    }

    private void redeemInvitation() {
        final JSONObject selected = invitation;
        if (selected == null || pairing) return;
        try {
            if (!credentials().getString("capture_credential", "").isEmpty()) {
                if (!pairedIdentity().equals(selected.getString("device_id"))) {
                    message("当前配对属于另一台机器人。请先明确断开并忘记旧设备，再导入邀请。"); return;
                }
                invitation = null;
                launch(null); return;
            }
        } catch (Exception e) { message("本地配对身份无效，请先忘记设备后重新配对"); return; }
        cancelled=false; pairing=true;
        worker.execute(() -> {
            try {
                PairChannel channel = new PairChannel(selected.getString("endpoint"), selected.getString("certificate_sha256"));
                JSONObject request = new JSONObject().put("invitation_id", selected.getString("invitation_id"))
                    .put("token", selected.getString("token")).put("device_id", selected.getString("device_id"))
                    .put("device_name", android.os.Build.MODEL);
                JSONObject result = channel.post("invite", request);
                byte[] pem = Base64.decode(result.getString("ca_certificate_base64"), Base64.DEFAULT);
                X509Certificate cert = (X509Certificate)java.security.cert.CertificateFactory.getInstance("X.509")
                    .generateCertificate(new ByteArrayInputStream(pem));
                if (!Arrays.equals(cert.getEncoded(), channel.certificate.getEncoded())
                        || !result.getString("wss_url").equals("wss://"+selected.getString("endpoint")+"/ws/teleop-capture"))
                    throw new IOException("邀请响应身份不一致");
                runOnUiThread(() -> { invitation=null; if (!cancelled) launch(result); });
            } catch (Exception e) { message("邀请连接失败，请在 Canvas 重新生成邀请："+e.getMessage()); }
            finally { pairing=false; }
        });
    }

    private void begin(String endpoint) {
        if (pairing) { message("已有配对申请，请先完成或取消后重试"); return; }
        if (!credentials().getString("capture_credential", "").isEmpty()) { message("已有配对，请先忘记设备并在卡片内撤销旧配对"); return; }
        cancelled=false; confirmed=false; pairing=true; confirm.setEnabled(false); confirm.setVisibility(android.view.View.GONE);
        message("正在申请连接，请在电脑端保持配对窗口打开…");
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
                message("请核对电脑端显示的配对码：\n"+fingerprint.replaceAll("(.{8})(?!$)","$1 ")+"\n两端确认后才能配对；不一致请取消。");
                runOnUiThread(() -> { confirm.setVisibility(android.view.View.VISIBLE); confirm.setEnabled(!cancelled); });
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
            } catch (Exception e) { message("pairing_window_closed".equals(e.getMessage()) ? "电脑端尚未允许配对或窗口已过期。请点击遥操卡片的“允许新设备配对”，再选择机器人。" : "连接失败，请检查网络和电脑端配对状态后重试。详情："+e.getMessage()); }
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
        if(request==42) { reconnect.setVisibility(credentials().getString("capture_credential", "").isEmpty() ? android.view.View.GONE : android.view.View.VISIBLE); message("采集连接已结束。若连接失败，请检查网络；身份变化或配对被撤销时需重新配对。可点击连接重试。"); }
    }
    @Override public void onPause() { cancelled=true; stopDiscovery(); super.onPause(); }
    @Override public void onDestroy() { cancelled=true; stopDiscovery(); worker.shutdownNow(); super.onDestroy(); }
}
