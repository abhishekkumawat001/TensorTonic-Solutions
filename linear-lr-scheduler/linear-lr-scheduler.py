def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    LR = 0
    eps = 1 * 10**-8

    if step < warmup_steps:
        LR = step * initial_lr / (warmup_steps + eps)
    elif warmup_steps <= step <= total_steps:
        LR = final_lr + ( initial_lr - final_lr ) *((total_steps - step)/(total_steps-warmup_steps +eps))
    else:
        LR = final_lr

    return LR