from collections import OrderedDict

def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import paddle
    import paddle.nn.functional as F
    import torch

    dont_transpose = ["emb.weight"]
    ignore = ["copy_mask"]

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")

    if hasattr(pytorch_state_dict, "state_dict"):
        config = pytorch_state_dict.config
        pytorch_state_dict = pytorch_state_dict.state_dict()

    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        if k in ignore:
            continue
        is_transpose = False

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = paddle.to_tensor(v.data.numpy())
    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "weights/enwik8-ppl1.65-6064-1024-RWKV-6-512-2022-03-25-21-05-13.pth", "weights/enwik8-ppl1.65-6064-1024-RWKV-6-512-2022-03-25-21-05-13.pdparams"
    )