package com.phanthymotus.capture;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Executes production parsing and TLS transport without creating an Activity. */
public final class InvitationContract {
    public static void main(String[] args) throws Exception {
        JSONObject test = new JSONObject(new String(System.in.readAllBytes(), StandardCharsets.UTF_8));
        JSONObject invitation = ConnectionInvitation.parse(test.getString("uri"));
        if (test.getString("mode").equals("parse")) {
            System.out.println(invitation.getString("endpoint"));
            return;
        }
        Class<?> channelClass = Class.forName("com.phanthymotus.capture.ConnectionActivity$PairChannel");
        Constructor<?> constructor = channelClass.getDeclaredConstructor(String.class, String.class);
        constructor.setAccessible(true);
        Object channel = constructor.newInstance(invitation.getString("endpoint"),
                                                 invitation.getString("certificate_sha256"));
        Method post = channelClass.getDeclaredMethod("post", String.class, JSONObject.class);
        post.setAccessible(true);
        JSONObject request = new JSONObject().put("invitation_id", invitation.getString("invitation_id"))
            .put("token", invitation.getString("token")).put("device_id", invitation.getString("device_id"))
            .put("device_name", "JVM contract fixture");
        JSONObject result = (JSONObject) post.invoke(channel, "invite", request);
        // Do not print pairing capabilities, keys or certificate contents.
        if (!result.getString("state").equals("approved")) throw new AssertionError("not approved");
        System.out.println("APPROVED");
    }
}
