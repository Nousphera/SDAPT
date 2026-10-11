from torch import Block, device, distributed, nn
import torch
import torch.nn.functional as F

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


def create_segmentation_input(x_tokens_list, use_n_blocks=1):
    """Return patch tokens to be used for a per-patch segmentation head.

    Expects each element in x_tokens_list to be a tuple (patch_tokens, class_token)
    where `patch_tokens` has shape (B, N_patches, D). This returns the patch
    tokens from the last block used.
    """
    intermediate_output = x_tokens_list[-use_n_blocks:]
    # take patch tokens from the last block
    patch_tokens = intermediate_output[-1][0]
    return patch_tokens.float()

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class SegmentationHead(nn.Module):
    def __init__(self, out_dim, use_n_blocks, num_classes, patch_size=14, img_size=518):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.h = self.w = img_size // patch_size  # e.g. 518//14 = 37

        # Project each patch token to class logits
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        # x_tokens_list: list of (patch_tokens, class_token) per block
        # patch_tokens shape: [B, N, D]  where N = H*W patches

        intermediate_output = x_tokens_list[-self.use_n_blocks:]

        # Concatenate patch tokens across the last n blocks along feature dim
        # Each patch_tokens: [B, N, D]  → cat → [B, N, D * use_n_blocks]
        patch_tokens = torch.cat(
            [patch_tokens for patch_tokens, _ in intermediate_output],
            dim=-1
        )

        # Classify each patch independently: [B, N, D*n] → [B, N, num_classes]
        logits = self.linear(patch_tokens)

        # Reshape to spatial grid: [B, H, W, num_classes] → [B, num_classes, H, W]
        B, N, C = logits.shape
        logits = logits.reshape(B, self.h, self.w, C)
        logits = logits.permute(0, 3, 1, 2)  # [B, num_classes, H, W]

        # Upsample from patch grid back to full image resolution
        logits = F.interpolate(
            logits,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False
        )  # → [B, num_classes, img_size, img_size]

        return logits


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict, optim_param_groups=[]):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)
        self.optim_param_groups = optim_param_groups

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }

# class KnnHead:
#     def __init__(self, k=5, temperature=0.07):
#         self.k = k
#         self.temperature = temperature
#         self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.labeled_features = []
#         self.labeled_targets = []

#     def _embed(self, encoder_client, encoder_server, x):
#         encoder_client.eval()
#         encoder_server.eval()
#         with torch.no_grad():
#             feats = encoder_client(x)
#             feats = encoder_server.get_intermediate_layers(
#                 feats, n=1, reshape=False, return_class_token=True, norm=True
#             )[0][1]
#             return F.normalize(feats, dim=1)

#     def fit(self, encoder_client, encoder_server, labeled_loader):
#         feats_all, targets_all = [], []
#         for x, y in labeled_loader:
#             feats_all.append(self._embed(encoder_client, encoder_server, x.to(self.device)))
#             targets_all.append(y.to(self.device))
#         self.labeled_features = torch.cat(feats_all)
#         self.labeled_targets = torch.cat(targets_all)

#     def predict_proba(self, encoder_client, encoder_server, x):
#         feats = self._embed(encoder_client, encoder_server, x.to(self.device))
#         sims = feats @ self.labeled_features.T                     # (B, N)

#         k = min(self.k, self.labeled_features.shape[0])
#         topk_sims, topk_idx = sims.topk(k, dim=1)                  # (B, k)
#         topk_labels = self.labeled_targets[topk_idx].float()       # (B, k)

#         w = torch.exp(topk_sims / self.temperature)                # (B, k)
#         score = (w * topk_labels).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12)
#         return score                                               # (B,) in [0, 1]

#     def predict(self, encoder_client, encoder_server, x, threshold=0.5):
#         return (self.predict_proba(encoder_client, encoder_server, x) >= threshold).long()
    
class KnnHead:
    def __init__(self, k=5):
        self.k = k
        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labeled_features = []  # stored labeled embeddings
        self.labeled_targets = []   # their labels
    
    def fit(self, encoder_client, encoder_server, labeled_loader):
        self.labeled_features = []
        self.labeled_targets = []
        encoder_client.eval()
        encoder_server.eval()
        with torch.no_grad():
            for x, y in labeled_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                feats = encoder_client(x)
                feats = encoder_server.get_intermediate_layers(feats, n=1, reshape=False, return_class_token=True, norm=True)[0][1]
                feats = F.normalize(feats, dim=1)
                self.labeled_features.append(feats)
                self.labeled_targets.append(y)
        
        # just stack and store — no gradient updates
        self.labeled_features = torch.cat(self.labeled_features)  # (N, D)
        self.labeled_targets = torch.cat(self.labeled_targets)    # (N,)
    
    def predict(self, encoder_client, encoder_server, x):
        encoder_client.eval()
        encoder_server.eval()
        with torch.no_grad():
            feats = encoder_client(x)
            feats = encoder_server.get_intermediate_layers(feats, n=1, reshape=False, return_class_token=True, norm=True)[0][1]
            feats = F.normalize(feats, dim=1)  # (B, D)
            
            # cosine similarity to every labeled sample
            sims = feats @ self.labeled_features.T  # (B, N)
            
            # find k nearest neighbors
            topk_sims, topk_idx = sims.topk(self.k, dim=1)  # (B, k)
            topk_labels = self.labeled_targets[topk_idx]     # (B, k)
            
            # majority vote
            predicted = torch.mode(topk_labels, dim=1).values
        
        return predicted