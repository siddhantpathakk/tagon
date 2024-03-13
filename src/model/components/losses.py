def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss
