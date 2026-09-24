import torch
import torch.nn as nn
import collections

from collections.abc import Callable, Iterable
from typing import Literal
from torch.distributions import Normal


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


# class CausalCrossAttention(nn.Module):
#     def __init__(self, z_dim, e_dim, hidden_dim=32):  # 32
#         super().__init__()
        
#         self.W_q = nn.Linear(z_dim, e_dim) 
#         self.W_k = nn.Linear(e_dim, e_dim)
        
#         self.attention_net = nn.Sequential(
#             nn.Linear(e_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, e_dim),
#             nn.Sigmoid()
#             # nn.Tanh()
#         )

#     def forward(self, z, e):
#         query = self.W_q(z)
#         key = self.W_k(e)
        
#         interaction = query * key 
#         attention_weights = self.attention_net(interaction)
        
#         e_tilde = attention_weights * e
        
#         return e_tilde, attention_weights


class CausalCrossAttention(nn.Module):
    def __init__(self, z_dim, e_dim, hidden_dim=32):
        super().__init__()
        self.W_q = nn.Linear(z_dim, e_dim)
        self.W_k = nn.Linear(e_dim, e_dim)

        # self.W_q = nn.Linear(z_dim, hidden_dim)
        # self.W_k = nn.Linear(e_dim, hidden_dim)
        
        self.attention_net = nn.Sequential(
            nn.Linear(e_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, e_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z, e):
        query = self.W_q(z)
        key = self.W_k(e)
        
        interaction = query * key 
        # interaction = torch.cat([query, key], dim=-1)
        
        delta = self.attention_net(interaction)
        
        e_tilde = e * delta
        
        return e_tilde, delta


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_cont
        The dimensionality of the continuous covariates
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_cont: int = 0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = n_cont + sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + self.n_cov * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int, cont: torch.Tensor | None = None):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        cont
            tensor of continuous covariates with shape ``(n_cont,)``

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many idxs useless.
        cont_list = [cont] if cont is not None else []
        cat_list = cat_list or []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        cov_list = cont_list + one_hot_cat_list
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            if (
                                x.device.type == "mps"
                            ):  # TODO: remove this when MPS supports for loop.
                                x = torch.cat(
                                    [(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0
                                )
                            else:
                                x = torch.cat(
                                    [layer(slice_x).unsqueeze(0) for slice_x in x], dim=0
                                )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                cov_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in cov_list
                                ]
                            else:
                                cov_list_layer = cov_list
                            x = torch.cat((x, *cov_list_layer), dim=-1)
                        x = layer(x)
        return x


class DecoderNB(nn.Module):
    """
    Decodes data from latent space to data space using Negative Binomial distribution.
    (Removed Zero-Inflation component compared to DecoderSCVI)
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma (Scale decoder) - 保持不变
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion (r decoder) - 保持不变
        # 负二项分布依然需要 dispersion 参数
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # [DELETE] 移除了 dropout decoder
        # self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):
        """
        Returns parameters for the Negative Binomial distribution.
        """
        # The decoder returns values for the parameters of the NB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        
        # [DELETE] 移除了 dropout 计算
        # px_dropout = self.px_dropout_decoder(px)
        
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale 
        
        # Dispersion logic remains the same
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        
        # 返回值中去掉了 px_dropout
        # 注意：如果你的下游 Loss 函数通过位置解包 (unpacking)，请检查是否需要调整接收变量
        return px_scale, px_r, px_rate


# def contrastive_consistency_loss(self, z_target, e_source):
#     """
#     InfoNCE Loss: 
#     让 e_source 和 它调节后的 e_mod (正样本) 距离拉近，
#     同时和 Batch 里其他的 e_other (负样本) 距离拉远。
#     """
#     batch_size = e_source.size(0)
#     e_mod, _ = self.crossattention(z_target, e_source.detach()) 

#     e_mod_norm = F.normalize(e_mod, dim=1)
#     e_source_norm = F.normalize(e_source.detach(), dim=1)
    
#     logits = torch.mm(e_mod_norm, e_source_norm.t())
    
#     # 4. InfoNCE Loss
#     labels = torch.arange(batch_size, device=e_source.device)
#     temperature = 0.05
    
#     loss = F.cross_entropy(logits / temperature, labels)
    
#     return loss

# def cycle_consistency_loss(
#         self, 
#         z_bg_ctrl: torch.Tensor,     # 来自 Control 的细胞背景 (Content Target)
#         z_t_trt: torch.Tensor,       # 来自 Treatment 的药物特征 (Style Target)
#         library_ctrl: torch.Tensor,  # Control 的 Library Size
#         batch_ctrl: torch.Tensor,    # Control 的 Batch Index
#         treat_name: str              # 当前药物名称
#     ) -> torch.Tensor:
        
#         # 1. 对齐 Batch Size (取两者较小值)
#         min_bs = min(z_bg_ctrl.size(0), z_t_trt.size(0))
        
#         # 截取数据
#         z_bg = z_bg_ctrl[:min_bs]      # 这是我们 Content 的真值 (Ground Truth)
#         z_t_target = z_t_trt[:min_bs]
#         lib = library_ctrl[:min_bs]
#         batch = batch_ctrl[:min_bs]
        
#         # 打乱 z_t_target (模拟随机注入药物)
#         rand_idx = torch.randperm(min_bs, device=z_bg.device)
#         z_t_target = z_t_target[rand_idx]

#         # 2. 融合 (Attention)
#         # z_bg 是 Query, z_t_target 是 Key
#         # z_t_fused 是我们 Style 的真值 (Ground Truth)
#         z_t_fused, _ = self.crossattention(z_bg, z_t_target)

#         # 3. 生成假细胞 (Decoder)
#         latent_input = torch.cat([z_bg, z_t_fused], dim=-1)
        
#         _, _, px_rate, _ = self.decoder(
#             self.dispersion,
#             latent_input,
#             lib,
#             batch
#         )
#         # _, _, px_rate = self.decoder(
#         #     self.dispersion,
#         #     latent_input,
#         #     lib,
#         #     batch
#         # )

#         x_fake = px_rate 

#         # 4. 重新编码 (Re-encode)
#         # 必须使用 log(x+1) 预处理
#         x_fake_log = torch.log(x_fake + 1)
        
#         bg_enc = self.treatment_background_encoders[treat_name]
#         te_enc = self.treatment_te_encoders[treat_name]
        
#         # 4.1 提取重构的 Content (z_bg)
#         _, _, z_bg_recon = bg_enc(x_fake_log, batch)
        
#         # 4.2 提取重构的 Style (z_t)
#         _, _, z_t_recon = te_enc(z_bg_recon)

#         # 5. 计算 Loss
        
#         # (A) 药物一致性 (Style Consistency)
#         # 生成的细胞，提取出的药物特征 z_t_recon 应该等于注入的 z_t_fused
#         loss_style = F.mse_loss(z_t_recon, z_t_fused.detach())
        
#         # (B) [补全] 内容一致性 (Content Consistency)
#         # 生成的细胞，提取出的背景身份 z_bg_recon 应该等于原来的 z_bg
#         # 逻辑：吃药不应该改变我是谁，只能改变我的状态
#         loss_content = F.mse_loss(z_bg_recon, z_bg.detach())
        
#         # 权重分配：
#         # 通常 style 权重可以大一点，因为这是你主要想学的变化
#         # content 权重是为了防止模型"飘"走
#         return loss_style + loss_content
    
# class CausalCrossAttention(nn.Module):
#     def __init__(self, z_dim, e_dim, hidden_dim=32):
#         super().__init__()
        
#         # 1. Query & Key Projections (保持不变)
#         self.W_q = nn.Linear(z_dim, e_dim) 
#         self.W_k = nn.Linear(e_dim, e_dim)
        
#         # 2. Attention Net: 决定"关注哪些特征" (方向)
#         # 保持原来的结构，负责特征选择 (Feature Selection)
#         self.attention_net = nn.Sequential(
#             nn.Linear(e_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, e_dim),
#             nn.Sigmoid() # 输出 0-1，作用是"过滤/筛选"
#         )

#         # 3. [新增] Intensity Gate: 决定"反应有多强" (模长)
#         # 专门根据细胞状态 z，预测一个全局放大系数
#         self.intensity_gate = nn.Sequential(
#             nn.Linear(z_dim, hidden_dim),
#             nn.LeakyReLU(),      # LeakyReLU 梯度流动更好
#             nn.Linear(hidden_dim, 1),
#             nn.Softplus()        # 关键！输出 > 0，且允许 > 1.0 (例如 2.5, 5.0)
#         )

#     def forward(self, z, e):
#         # --- 步骤 1 & 2: 投影 ---
#         query = self.W_q(z)  # [batch, e_dim]
#         key = self.W_k(e)    # [batch, e_dim]
        
#         # --- 步骤 3: 交互 (Interaction) ---
#         # 保持你原来的设计，捕捉 z 和 e 的对齐程度
#         interaction = query * key 
        
#         # --- 步骤 4: 计算注意力权重 (Selection) ---
#         # 这一步决定了"哪些药物特征生效"
#         # 范围 [0, 1]
#         selection_weights = self.attention_net(interaction)
        
#         # --- 步骤 5: [新增] 计算反应强度 (Magnitude) ---
#         # 这一步决定了"细胞的反应有多剧烈"
#         # 范围 [0, +inf)
#         # 强反应细胞 z 会预测出一个大的 scale
#         intensity_scale = self.intensity_gate(z) 
        
#         # --- 步骤 6: 应用 Attention ---
#         # 逻辑：原始 e * 特征筛选 * 强度放大
#         # 比如：e * 0.9 (筛选) * 3.0 (放大) = 2.7倍增强
#         e_tilde = e * selection_weights * intensity_scale
        
#         return e_tilde, selection_weights
    
# def hsic_loss(self, x, y):
#     # HSIC (Hilbert-Schmidt Independence Criterion) 实现
#     # 适用于小 Batch，计算两个分布的独立性
#     m = x.size(0)
#     # 计算核矩阵 (Gaussian Kernel)
#     def compute_kernel(u, v):
#         dist = torch.cdist(u, v, p=2).pow(2)
#         # 动态计算带宽 sigma
#         sigma = dist.median() + 1e-5 
#         return torch.exp(-dist / sigma)
    
#     K_x = compute_kernel(x, x)
#     K_y = compute_kernel(y, y)
    
#     # 中心矩阵 H
#     H = torch.eye(m, device=x.device) - (1.0 / m) * torch.ones((m, m), device=x.device)
    
#     # HSIC = tr(K_x * H * K_y * H) / (m-1)^2
#     hsic = torch.trace(torch.mm(torch.mm(K_x, H), torch.mm(K_y, H))) / ((m - 1) ** 2)
#     return hsic

# def counterfactual_consistency_loss(self, z, e):
#     e_source = e
#     if e_source.size(0) < 2: return torch.tensor(0.0, device=z.device)

#     batch_size = e_source.size(0)
#     rand_indices = torch.randperm(z.size(0), device=z.device)
#     z_target = z[rand_indices[:batch_size]]
    
#     # 生成反事实特征
#     e_counterfactual, _ = self.crossattention(z_target, e_source.detach())
    
#     # [核心改进] 
#     # 目标：e_counterfactual (结果) 应该与 z_target (注入的随机噪声) 相互独立
#     # 如果 HSIC 接近 0，说明 Attention 成功忽略了 z_target 的干扰
#     loss = self.hsic_loss(e_counterfactual, z_target)
    
#     return loss

# @staticmethod
# def reconstruction_loss(
#         x: torch.Tensor,
#         px_rate: torch.Tensor,
#         px_r: torch.Tensor,
# ) -> torch.Tensor:
#     if x.shape[0] != px_rate.shape[0]:
#         print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
#     recon_loss = (
#         -NegativeBinomial(mu=px_rate, theta=px_r)
#         .log_prob(x)
#         .sum(dim=-1)
#     )
#     return recon_loss

# @auto_move_data
# def generative(
#         self,
#         z_bg: torch.Tensor,
#         z_t: torch.Tensor,
#         library: torch.Tensor,
#         batch_index: List[int],
# ) -> Dict[str, Dict[str, torch.Tensor]]:
#     z_t, _ = self.crossattention(z_bg, z_t)
#     latent = torch.cat([z_bg, z_t], dim=-1)
#     px_scale, px_r, px_rate = self.decoder(
#         self.dispersion,
#         latent,
#         library,
#         batch_index,
#     )
#     px_r = torch.exp(self.px_r)
#     return {
#         'px_scale': px_scale,
#         'px_r': px_r,
#         'px_rate': px_rate,
#     }
