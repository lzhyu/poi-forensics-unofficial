import torch
import numpy as np
# copied from: https://github.com/google-research/google-research/blob/master/supcon/losses.py

def contrastive_loss(features: torch.Tensor,
                     labels: torch.LongTensor,
                     device, 
                     temperature=1.0,
                     scale_by_temperature=True):

    # features: B X feature_dim
    # labels: B X 1
    batch_size = features.size()[0]

    # get labels
    # mask should be B X B M[i][j] = (i.id==j.id), correct
    labels_x = labels.repeat((1, batch_size))
    labels_y = labels_x.t()
    diagonal_mask = (labels_x==labels_y).to(torch.int)
    
    # B X feature_dim X 1, correct
    anchor_features = torch.unsqueeze(features, 2)
    # anchor_features[i] = feature_i
    # B X B X feature_dim
    # features[i] = [features], features[i][j] = features_j, correct
    all_global_features = torch.unsqueeze(features, 0)
    all_global_features = all_global_features.repeat((batch_size, 1, 1))    

    # Generate `logits`, the tensor of (temperature-scaled) dot products of the
    # anchor features with all features. It has shape
    # [local_batch_size * num_anchor_views, global_batch_size * num_views]. To
    # improve numerical stability, subtract out the largest |logits| element in
    # each row from all elements in that row. Since |logits| is only ever used as
    # a ratio of exponentials of |logits| values, this subtraction does not change
    # the results correctness. A stop_gradient() is needed because this change is
    # just for numerical precision.

    # get logits, (B X feature_dim) X (feature_dim X 1) = B X 1
    # logits B X B, logits[i][j] = dot(features[i],features[j])

    # cosin distance
    # logits = torch.bmm(
    #     all_global_features, anchor_features)
    # logits = torch.squeeze(logits, dim=2)

    # norm distance, NOTE:seems correct
    anchor_features = torch.squeeze(anchor_features, dim=2).unsqueeze(1).repeat((1,batch_size,1))
    # anchor_features[i][j] = [feature_i]
    logits = -torch.norm(anchor_features-all_global_features, p=2, dim=2)

    logits = logits / temperature
    logits = logits - torch.max(logits.detach(), dim=0, keepdim=True)[0]
    # subtract the max value in a row
    exp_logits = logits.exp()

    # NOTE: record nearest
    # should have B X 1, record the index of nearest
    # nearest_indices = torch.topk(logits, 2, dim=1, largest=True)[1]
    # compute the similarity between zero and array
    similarity_zero = logits[0]
    # [B]
    
    # record distance metrix

    # The following masks are all tiled by the number of views, i.e., they have
    # shape [local_batch_size * num_anchor_views, global_batch_size * num_views].

    # self_diagonal_mask: B X B diagonal
    self_diagonal_mask = torch.eye(batch_size).to(device)
    positives_mask = diagonal_mask - self_diagonal_mask
    negatives_mask = 1 - positives_mask - self_diagonal_mask
    num_positives_per_row = positives_mask.sum(dim=1)
    
    assert((num_positives_per_row>=1).all())
    # reduce sum
    denominator = torch.sum(
        exp_logits * negatives_mask, dim=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdims=True)

    # Note that num_positives_per_row can be zero only if 1 view is used. The
    # various tf.math.divide_no_nan() calls below are to handle this case.
    log_probs = (logits - torch.log(denominator))*positives_mask
    log_probs = torch.sum(log_probs, axis=1)

    # FIXME: check the nearest neighbor

    loss = -log_probs
    if scale_by_temperature:
        loss *= temperature

    loss = loss.mean()

    return loss, similarity_zero# nearest_indices

if __name__=='__main__':
    features = torch.randn((8,10))
    labels = torch.Tensor(np.array([[1],[1],[2],[2],[3],[3],[4],[4]]))
    print(features)
    device = 'cpu'
    print(contrastive_loss(features, labels, device))