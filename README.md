# Kp docker 介紹

## 1. Toolchain Docker Deployment

### 1.1 TAGS List

```bash
curl https://registry.hub.docker.com/v2/repositories/kneron/toolchain/tags | yq -p json -o yaml | grep ' name: '
```

```
    name: latest
    name: v0.23.1
    name: "720"
    name: "520"
    name: base-20230922
    name: base
    name: v0.23.0
    name: se_20230926
    name: v0.22.0
    name: v0.21.0
    name: base-20230321
    name: v0.20.2
    name: v0.20.1
    name: v0.20.0
    name: v0.19.0
    name: v0.18.2
    name: v0.18.1
    name: base-20220804
    name: v0.18.0_s0
    name: v0.18.0
    name: base-20220622
    name: base-20220615
    name: base-20220609
    name: base-20220602
    name: base-20220526
```
