import time

def remaining_time(t_0, epoch, max_epoch, batch, max_batch):
    if epoch == 0 and batch == 0:
        return "?"

    delta_t = time.time() - t_0
    mean_batch_time = delta_t / (batch + max_batch * epoch)

    rem_time =  mean_batch_time * ((max_epoch - epoch - 1) * max_batch + max_batch - batch)

    return time.strftime('%H:%M:%S', time.gmtime(rem_time))