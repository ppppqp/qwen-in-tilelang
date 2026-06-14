def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen3-8b":
        return "Qwen/Qwen3-8B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-0.6b":
        return "Qwen/Qwen3-0.6B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-1.7b":
        return "Qwen/Qwen3-1.7B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-4b":
        return "Qwen/Qwen3-4B-MLX-4bit"
    elif lower_shortcut_name in ("qwen3-30b-a3b", "qwen3-moe-30b-a3b"):
        return "Qwen/Qwen3-30B-A3B-MLX-4bit"
    else:
        return shortcut_name
