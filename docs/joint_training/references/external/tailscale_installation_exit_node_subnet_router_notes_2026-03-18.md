# Tailscale 安装记录与 Exit Node / Subnet Router 说明

日期：2026-03-18

本文档记录了在 `pm-1782` 上实际执行的 Tailscale 安装、登录和联通性排查过程，并整理了 `exit node` 与 `subnet router` 的用途、优缺点以及它们与代理软件的关系。它作为一次真实操作的运维参考，供后续复用。

## 1. 环境与目标

- 主机名：`pm-1782`
- 系统：`Ubuntu 22.04.5 LTS`
- 架构：`x86_64`
- 目标：在这台服务器上安装 Tailscale，并加入与用户笔记本及另一台服务器相同的 tailnet
- 登录完成后的节点信息：
  - Tailscale IPv4：`100.108.27.88`
  - Tailscale IPv6：`fd7a:115c:a1e0::bc3a:1b58`
  - MagicDNS 名称：`pm-1782.tail0a8c6f.ts.net`
  - 所属 tailnet：`geekmercer@gmail.com`

## 2. 在 `pm-1782` 上的实际安装过程

### 2.1 安装前检查

安装前执行了以下检查：

```bash
uname -a
cat /etc/os-release
command -v tailscale || true
command -v tailscaled || true
id
```

实际结论：

- 机器运行的是 `Ubuntu 22.04.5 LTS`
- 当时系统里还没有安装 `tailscale` 和 `tailscaled`
- 当前操作用户是 `root`，因此不需要额外切换权限

### 2.2 从官方源安装 Tailscale

实际安装命令：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

该安装脚本实际完成了以下动作：

1. 为 Ubuntu Jammy 添加 Tailscale 的 APT 仓库
2. 安装 Tailscale 软件仓库签名 key
3. 执行 `apt-get update`
4. 安装以下包：
   - `tailscale`
   - `tailscale-archive-keyring`

安装后的版本为：

```text
1.94.2
```

### 2.3 启动并检查 `tailscaled`

安装完成后，执行了：

```bash
systemctl start tailscaled
systemctl is-enabled tailscaled
systemctl is-active tailscaled
```

检查结果：

- `tailscaled` 已被设置为开机自启
- `tailscaled` 当前处于运行状态

### 2.4 执行登录并加入 tailnet

随后执行：

```bash
tailscale up
```

在用户尚未完成浏览器认证之前，`tailscale status --json` 的状态是：

- `BackendState: NeedsLogin`
- 同时返回一次性 `AuthURL`

用户随后在笔记本上打开登录链接并批准节点，这台服务器正式加入现有 tailnet。

### 2.5 登录后校验节点状态

登录完成后，执行了：

```bash
tailscale status --json
tailscale ip -4
tailscale version
```

关键结果如下：

- `BackendState: Running`
- IPv4：`100.108.27.88`
- IPv6：`fd7a:115c:a1e0::bc3a:1b58`
- MagicDNS 后缀：`tail0a8c6f.ts.net`
- tailnet 名称：`geekmercer@gmail.com`
- 当前可见的 peer 包括：
  - `Alex`（`100.99.53.104`，Windows）
  - `iZ0xi0n57k17xrzgxh0me2Z`（`100.67.178.53`，Linux）

## 3. 联通性检查，以及 DERP 与直连的区别

### 3.1 初始 `tailscale ping` 为什么走 DERP

最初对两个现有节点执行了：

```bash
tailscale ping -c 2 100.99.53.104
tailscale ping -c 2 100.67.178.53
```

一开始两条路径都显示为 DERP 中继：

- `Alex` 通过 `DERP(nue)`
- `iZ0xi0n57k17xrzgxh0me2Z` 通过 `DERP(iad)`

这并不代表配置错误。Tailscale 常见的行为是：

1. 先通过 DERP 建立一个可用的初始路径
2. 同时在后台完成 NAT 打洞、endpoint 学习和直连探测
3. 一旦发现可用的公网 UDP endpoint，就切换为直连

也就是说，第一次看到 DERP 不应立即判定为“只能中继，无法直连”。

### 3.2 用 `netcheck` 和本机防火墙判断服务器是否具备直连条件

为判断 `pm-1782` 本机是否具备直连能力，执行了以下检查：

```bash
tailscale netcheck
ss -lunp
ufw status verbose || true
nft list ruleset || true
iptables -S || true
iptables -t nat -S || true
```

关键观察点如下：

- `tailscale netcheck` 报告：
  - `UDP: true`
  - `IPv4: yes, 125.122.157.32:44792`
  - `MappingVariesByDestIP: false`
- `tailscaled` 正在监听 UDP `41641`
- 本机 `nftables/iptables` 中已经存在 Tailscale 自动写入的放行规则，允许 UDP `41641`
- `tailscale netcheck` 的日志里还能看到：

```text
tshttpproxy: using proxy "http://127.0.0.1:7891"
```

这说明这台服务器本身存在一个本地代理，但它影响的是控制面 HTTPS 请求，不妨碍 Tailscale 数据面继续走 UDP 打洞和直连。

从这些信号可以得出一个重要结论：

- `pm-1782` 本身并不是一个“只能走中继”的网络环境
- 这台机器具备建立直连的基础条件

### 3.3 从日志里确认 endpoint 学习是否成功

进一步查看 `tailscaled` 日志后，出现了下面这些关键信息：

```text
magicsock: endpoints changed: 125.122.157.32:41641 (stun), 10.0.1.4:41641 (local), 172.17.0.1:41641 (local)
magicsock: disco: node [hhoY8] ... now using 124.127.58.150:39043
magicsock: disco: node [XmvuC] ... now using 47.252.33.133:41641
```

这些日志表示：

1. `pm-1782` 已经确认了自己的可用公网 IPv4 endpoint
2. Tailscale 已经从两个 peer 处学到了可用于直连的公网 UDP endpoint

这一步非常关键，因为它说明问题不在“服务端完全不会打洞”，而在于“直连建立需要一点时间，第一次探测窗口太早”

### 3.4 最终验证：两条链路都已经切成直连

在额外做了几轮探测之后，最终观察到了直连结果：

```bash
tailscale ping -c 10 100.99.53.104
tailscale ping -c 5 100.67.178.53
```

最终结果如下：

- `Alex (100.99.53.104)`：

```text
pong from alex (100.99.53.104) via 124.127.58.150:26025 in 42ms
```

- `iZ0xi0n57k17xrzgxh0me2Z (100.67.178.53)`：

```text
pong from iz0xi0n57k17xrzgxh0me2z (100.67.178.53) via 47.252.33.133:41641 in 213ms
```

结论是：

- 这台服务器最终已经能和两个现有节点建立直连
- 之前看到 DERP，只是刚入网时的暂时状态，不是永久性故障

### 3.5 这次排查沉淀出的运维经验

- 刚加入 tailnet 的新节点，第一次 `tailscale ping` 走 DERP 很常见
- 不要只看第一次 `tailscale ping` 就判断“直连失败”
- 更可靠的排查顺序应该是：
  1. 看 `tailscale netcheck`
  2. 看本机 UDP 监听与防火墙
  3. 看 `tailscaled` 日志是否学到了 peer endpoint
  4. 再做多几次 `tailscale ping`

## 4. 什么是 Exit Node

`exit node` 的作用是：让某台 Tailscale 节点充当其他设备的默认互联网出口。

当一台客户端启用了 exit node 后，流量路径会变成：

1. 客户端先把公网流量加密送进 Tailscale 隧道
2. 这些流量被送到被选中的 exit node
3. exit node 再以自己的公网 IP 把流量转发到互联网

从外部网站的视角来看，这台客户端就像是“从 exit node 所在位置和 IP 上网”。

### 4.1 Exit Node 的典型用途

- 笔记本需要借用某台服务器的公网 IP 出网
- 某些系统只对白名单 IP 开放访问
- 在酒店、机场、咖啡馆等不可信网络中，希望统一从自己的服务器安全出网
- 多台设备希望共享同一个出口 IP，统一日志、审计或地理位置
- 需要整机走另一台可信设备，而不是单独给每个应用配置代理

### 4.2 Exit Node 的优点

- 出口 IP 固定
- 对整机生效，使用体验简单
- 不需要针对浏览器或单个应用分别配置代理
- 对“我只想让整台机器从那台服务器出去”这种需求非常直接

### 4.3 Exit Node 的代价与限制

- 所有公网流量都依赖出口节点的带宽、延迟和稳定性
- 出口节点的维护者可以看到访问目标的元数据，例如目标 IP 和域名
- 如果客户端还想继续访问本地局域网，通常还需要配合 `allow LAN access`
- Exit node 改变的是整机默认路由，不是精细的按域名、按应用、按地区分流工具

## 5. Exit Node 和代理软件是什么关系

Exit node 与代理软件解决的不是同一层的问题。

### 5.1 Exit Node 的定位

- 工作在网络路由层
- 通常对整台设备生效
- 最适合“让这台机器整体从另一台机器出去”这种需求

### 5.2 代理软件的定位

- 可以工作在应用层、TUN 层或透明代理层
- 适合做更细的流量策略，例如：
  - 按域名分流
  - 按应用分流
  - 不同地区节点切换
  - 多上游容灾
  - 广告过滤、规则引擎、流量分类

### 5.3 什么情况下 Exit Node 可以替代本地代理

如果你的目标只是：

- “让我的笔记本看起来像从这台服务器公网 IP 出去”
- “让整台设备都走我自己的服务器出网”

那么 Exit Node 往往可以替代客户端本地代理。

也就是说，在这种场景下，客户端可以不再开本地全局代理，直接把出口交给 Tailscale。

### 5.4 什么情况下代理软件依然有必要

如果你的真实需求是：

- 按域名做精细分流
- 某些应用走代理、某些应用不走
- 多个国家或地区出口切换
- 自动选路、故障切换
- 广告过滤或更细的策略控制

那么 Exit Node 无法替代代理软件，代理依然有存在价值。

### 5.5 当“服务器和笔记本都在用代理软件”时应该怎么理解

这是最容易把路由搞复杂的情况。

常见的三种设计是：

1. **Tailscale 只负责私网互联**
   - 笔记本和服务器只把 Tailscale 用作私网地址层
   - 公网访问继续各走各的代理配置
   - 这是最简单、最不容易出问题的设计

2. **服务器提供 Exit Node，笔记本不用本地代理**
   - 笔记本通过 Tailscale 把默认出口交给服务器
   - 笔记本本地代理临时关闭
   - 适合“我只想借服务器公网 IP 出网”的场景

3. **服务器既是 Exit Node，又在服务器侧继续做代理**
   - 笔记本先走 Tailscale 到服务器
   - 然后服务器再决定是否用自己的 TUN/透明代理继续往外转
   - 这个方案可以工作，但必须有意识地设计，复杂度明显更高

最重要的运维建议是：

- 不要在同一台客户端上同时叠加两个“全局默认路由接管者”，除非你非常清楚自己要的效果
- 笔记本上的全局代理和 Exit Node 同时开启时，最常见的问题不是“不能用”，而是“很难判断到底谁在生效”

## 6. 什么是 Subnet Router

`subnet router` 的作用是：让某台 Tailscale 节点把自己后面的一段私有网络发布给整个 tailnet。

与 Exit Node 不同，Subnet Router 转发的不是默认公网流量，而是特定的私网网段。

举例：

- 某台服务器能访问一个 VPC 内网 `10.0.0.0/24`
- 这台服务器把 `10.0.0.0/24` 广播给 tailnet
- 其他 Tailscale 设备就能通过它访问这个内网，而不需要在每一台内网主机上都安装 Tailscale

### 6.1 Subnet Router 的典型用途

- 从笔记本访问家里、办公室或机房里的整个内网
- 从笔记本访问某个云 VPC 的私网地址
- 访问没有安装 Tailscale 的设备，例如打印机、NAS、内部数据库、老旧服务
- 用一台中间机器把一整段私有网桥接进 tailnet

### 6.2 Subnet Router 的优点

- 可以覆盖没有安装 Tailscale 的设备
- 不需要给内网里的每台机器逐台部署客户端
- 对明确、稳定的 CIDR 网段非常合适

### 6.3 Subnet Router 的代价与限制

- 广播出来的网段不能和客户端本地已有路由冲突
- 这些路由往往需要在 Tailscale 管理后台审批
- 路由节点本身需要正确开启 IP forwarding 和配套防火墙/NAT 规则
- 相比“每台机器都装 Tailscale”，引入子网路由会增加运维复杂度

### 6.4 Subnet Router 与 Exit Node 的根本区别

- `exit node`：转发默认互联网流量，也就是 `0.0.0.0/0` 与 `::/0`
- `subnet router`：只转发指定私网段，例如 `10.0.0.0/24` 或 `192.168.0.0/24`

可以用一句话来记：

- 如果需求是“让这台笔记本像从那台服务器上网”，选 Exit Node
- 如果需求是“让这台笔记本能访问那台服务器后面的一整段私网”，选 Subnet Router

## 7. 基于当前 `tailscale 1.94.2` CLI 的命令说明

在这台服务器上查看 `tailscale --help` 后，当前版本支持 `tailscale set`。对于后续运维，它比 `tailscale up` 更安全，因为它只改你显式指定的设置，不要求你重新把所有现有参数都写全。

### 7.1 把当前节点声明为 Exit Node

```bash
tailscale set --advertise-exit-node
```

### 7.2 在客户端上使用某个 Exit Node

```bash
tailscale set --exit-node=pm-1782 --exit-node-allow-lan-access=true
```

### 7.3 广播一个子网路由

```bash
tailscale set --advertise-routes=10.0.0.0/24
```

### 7.4 清空已经广播的子网路由

```bash
tailscale set --advertise-routes=
```

### 7.5 关于 `tailscale up` 的补充说明

`tailscale up` 在当前版本依然可用，但它更适合“首次拉起”和“完整指定一整套配置”的场景。因为一旦带参数调用，它会要求你给出完整的目标配置集。相比之下，`tailscale set` 更适合后续小范围调整。

## 8. 对当前环境的实际建议

结合这次真实操作，当前环境的建议如下：

- 这台服务器已经能够正常加入 tailnet
- 这台服务器具备直连能力，并且最终已经和两个现有节点建立了直连
- 这台服务器虽然存在本地代理参与部分控制面流量，但没有阻止 Tailscale 数据面建立 UDP 直连

因此：

- 如果需求只是“笔记本与服务器私网互联”，那就维持当前普通 Tailscale 模式，不必开启 Exit Node 或 Subnet Router
- 如果需求变成“让笔记本统一从这台服务器公网 IP 出网”，再考虑把这台服务器声明为 Exit Node
- 如果你真正想要的是复杂分流策略、规则引擎或多出口切换，那就继续保留代理软件，把 Tailscale 当作私网互联层，而不是完全替代代理
- 只有当某台机器后面还挂着一整段没有装 Tailscale 的私网时，才值得启用 Subnet Router

## 9. Tailscale 内网 AI 服务与本机 Mihomo 订阅规则的关系

这一节专门记录 `http://100.67.178.53:8080` 这种“位于 tailnet 中另一台服务器上的 AI 服务”与本机代理配置之间的真实关系。

### 9.1 被访问的目标是什么

本次被测试的目标服务是：

```text
http://100.67.178.53:8080
```

它位于同一 tailnet 中的另一台服务器上，属于 Tailscale 内网地址，不是公网地址。

### 9.2 本机代理软件的实际运行方式

这台服务器上运行的是 `Mihomo Meta`，其运行配置显示：

- `mode: rule`
- `mixed-port: 7890`
- `port: 7891`
- `socks-port: 7892`

同时，这个 shell 环境中默认存在：

```text
HTTP_PROXY=http://127.0.0.1:7891
HTTPS_PROXY=http://127.0.0.1:7891
ALL_PROXY=socks5h://127.0.0.1:7892
NO_PROXY=localhost,127.0.0.1,::1
```

这意味着：

1. 普通命令行程序如果直接调用 HTTP/HTTPS，请求默认会先发给本机 Mihomo
2. 但是否真正经过远端代理节点，不取决于环境变量本身，而取决于 Mihomo 最终命中的规则

### 9.3 两个订阅对 Tailscale 地址的规则判断是一致的

本机当前存在两个订阅：

- `ID 1`：`/root/clashctl/resources/profiles/1.yaml`
- `ID 2`：`/root/clashctl/resources/profiles/2.yaml`

实际检查结果是：

- `ID 1` 包含：

```text
IP-CIDR,100.64.0.0/10,🎯 全球直连,no-resolve
```

- `ID 2` 包含：

```text
IP-CIDR,100.64.0.0/10,DIRECT
```

而 `100.67.178.53` 正好落在 `100.64.0.0/10` 这个 Tailscale 常见地址段内。

因此，对这两个订阅来说：

- `100.67.178.53:8080` 都会命中直连规则
- 不会落到代理组，例如 `Cyber Paws` 或其他远端节点选择组

### 9.4 当前实际生效的是哪个订阅

`/root/clashctl/resources/profiles.yaml` 显示当前正在使用的是：

```text
use: 2
```

也就是说，当前运行态实际使用的是 `ID 2`，不是早期文档里提到的旧状态。

不过由于 `ID 1` 和 `ID 2` 对 `100.64.0.0/10` 都给了直连规则，所以这个结论对两份订阅都成立。

### 9.5 实际访问结果：既能直连，也不会和 Mihomo 规则冲突

本次对 `http://100.67.178.53:8080` 做了两种访问测试：

1. 直接绕过环境代理访问：

```bash
curl --noproxy '*' http://100.67.178.53:8080
```

结果：

- 成功建立到 `100.67.178.53:8080` 的 TCP 连接
- 返回 `HTTP/1.1 200 OK`
- 首页内容显示为 `Sub2API - AI API Gateway`

2. 在保留当前 `HTTP_PROXY/HTTPS_PROXY` 环境变量的情况下访问：

```bash
curl http://100.67.178.53:8080
```

结果同样返回 `200 OK`。

这说明当前环境下的真实行为是：

1. 请求可能先发给本机 Mihomo 监听端口
2. Mihomo 根据规则识别出目标 IP 位于 `100.64.0.0/10`
3. Mihomo 对该连接做 `DIRECT`
4. 流量最终通过本机 Tailscale 路由到另一台 tailnet 服务器

换句话说，当前配置下并不存在“本机代理软件把这条 Tailscale 内网流量错误地送去远端代理节点”的冲突。

### 9.6 这三层关系应该怎么理解

对 `http://100.67.178.53:8080` 这类地址来说，链路关系可以拆成三层：

1. **应用层**
   - 应用程序、SDK、`curl` 或浏览器发起请求

2. **本机代理判断层**
   - 如果程序继承了 `HTTP_PROXY/HTTPS_PROXY`，请求会先进入本机 Mihomo
   - Mihomo 依据订阅规则判断这个目标应该 `DIRECT` 还是走代理组

3. **底层网络层**
   - 一旦 Mihomo 选择 `DIRECT`，真正的底层连通由系统路由负责
   - 对于 `100.67.178.53` 这样的 Tailscale IP，系统会把它交给 Tailscale 网络

所以从职责划分上说：

- Tailscale 负责“这条内网地址能不能到达”
- Mihomo 负责“这条连接是否应该走远端代理节点”

当前这台服务器上的配置结论是：

- Tailscale 可达
- Mihomo 对该地址判断为直连
- 两者并不冲突

### 9.7 在什么情况下才可能发生冲突

虽然当前配置没有冲突，但以下场景可能导致后续出现问题：

1. **换到另一台机器时，规则未必一样**
   - 这里只验证了 `pm-1782` 本机的 Mihomo 订阅
   - 其他机器若没有 `100.64.0.0/10,DIRECT`，就不能直接套用这个结论

2. **程序绕过了本机 Mihomo，转而使用一个外部代理**
   - 如果某个 SDK 或程序不是把流量交给本机 `127.0.0.1:7891/7892`
   - 而是直接交给一个公网代理服务器
   - 那个公网代理通常并不知道怎么访问 `100.67.178.53`

3. **目标地址不再是 Tailscale IP**
   - 例如未来把 `base_url` 改成一个公网域名或公网 IP
   - 那就要重新按域名/IP 规则判断，不再自动享受 `100.64.0.0/10` 直连规则

4. **把规则改坏**
   - 如果未来订阅更新或自定义规则覆盖掉了 `100.64.0.0/10,DIRECT`
   - 这类 tailnet 地址就可能落到代理组

### 9.8 对使用 AI API 的实际影响

如果这个站点本身部署的是 AI 网关，并向客户端提供：

- `api key`
- `base_url=http://100.67.178.53:8080`

那么在这台服务器上，当前配置下可以这样理解：

- 客户端请求到这个 `base_url` 时，不会被远端代理节点接管
- 这条链路最终走的是 Tailscale 内网直连
- 至于该 AI 网关后端是否还需要代理去访问 OpenAI、Anthropic 或其他模型服务，那是**服务端自己的上游出网问题**，与这里“客户端到网关”的这段链路是两回事

因此，本机当前环境下的真实结论是：

- 访问这个 Tailscale 内网 AI 服务不会与 Mihomo 的两个订阅规则冲突
- 即便命令行进程继承了本机 `HTTP_PROXY/HTTPS_PROXY`，最终也会因为订阅规则命中 `100.64.0.0/10` 而走直连
- 如果只是为了减少一层本地代理转发和调试复杂度，可以额外把 Tailscale 地址加入 `NO_PROXY`，但这不是当前可用性的必要条件
