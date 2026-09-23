package com.phanthymotus.capture;

import android.util.Base64;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Arrays;
import org.json.JSONObject;

/** The URI fragment is an enrollment capability, never a motion authorization. */
final class ConnectionInvitation {
    static JSONObject parse(String value) throws Exception {
        if (value == null || value.length() > 4096) throw new IllegalArgumentException();
        URI uri = new URI(value);
        if (!"motus-teleop".equals(uri.getScheme()) || !"connect".equals(uri.getHost())
            || uri.getUserInfo()!=null || uri.getPort()!=-1 || uri.getRawQuery()!=null
            || (uri.getPath()!=null && !uri.getPath().isEmpty())
            || uri.getRawFragment()==null || !uri.getRawFragment().matches("[A-Za-z0-9_-]{1,3500}"))
            throw new IllegalArgumentException();
        byte[] bytes = Base64.decode(uri.getRawFragment(), Base64.URL_SAFE|Base64.NO_WRAP|Base64.NO_PADDING);
        JSONObject data = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
        HashSet<String> keys = new HashSet<>();
        java.util.Iterator<String> it = data.keys();
        while(it.hasNext()) keys.add(it.next());
        if (!keys.equals(new HashSet<>(Arrays.asList("schema", "invitation_id", "token", "device_id", "certificate_sha256", "endpoint")))
            || !"motus.teleop.invitation.v1".equals(data.getString("schema"))
            || !data.getString("device_id").matches("[0-9a-f]{64}")
            || !data.getString("device_id").equals(data.getString("certificate_sha256"))
            || !data.getString("token").matches("[A-Za-z0-9_-]{32,128}")
            || !java.util.UUID.fromString(data.getString("invitation_id")).toString().equals(data.getString("invitation_id")))
            throw new IllegalArgumentException();
        String endpoint = data.getString("endpoint");
        URI target = new URI("https://"+endpoint);
        if(endpoint.length()>255 || target.getHost()==null || target.getUserInfo()!=null
            || target.getPort()<1 || target.getPort()>65535 || target.getRawQuery()!=null
            || target.getRawFragment()!=null || !target.getPath().isEmpty()) throw new IllegalArgumentException();
        return data;
    }
}
