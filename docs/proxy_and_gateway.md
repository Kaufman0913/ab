# Proxy and Gateway

There are two concepts that might seem confusing in the sandbox architecture. These are the proxy and the gateway.

The basis of these is that sandboxes need a way to access the internet. However, we don't always want sandboxes to have unrestricted access to the internet. This is insecure and dangerous.

## Proxy

To mitigate this, we make it so that some sandboxes only have access to a *sandbox proxy*. The sandbox proxy is a special Docker container. This means that it runs inside Docker, not the host operating system. All other restricted sandboxes are allowed to connect to it (typically at `http://sandbox_proxy/`). They can send whatever requests they want and the proxy will forward them to the *gateway*.

## Gateway

The gateway is a process that runs on an actual computer. It is another HTTP server. This process has full access to the entire Internet. You can implement whatever you want in the gateway, whatever you feel is safe for the sandbox to be able to access.

<br>
<br>
<br>

# In Practice

In Ridges, we want sandboxes to have access to AI tools (inference, embedding, etc.) but no access to anything else.

We also don't want the sandboxes to be able to communicate with our inference providers (Chutes, Targon, etc.) directly, as they would need our API keys in order to do that.

Instead, sandboxes make requests to `http://sandbox_proxy/api/inference` (you should always use the `SANDBOX_PROXY_URL` in case we change this in the future). This resolves to a Docker container known as the sandbox proxy.

The sandbox proxy then verifies that the request is going to an appropriate URL (such as `/api/inference` or `/api/embedding`, but not `/api/foo`), and then forwards it to the gateway.

The gateway (which runs in a physical server, not a Docker container), handles appropriate requests and forwards them to Chutes or Targon, after performing some checks (correct data types, valid run_id, etc.).