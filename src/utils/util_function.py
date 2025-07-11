def recursive_dict_update(original, update_data):
    """
    递归更新字典，保留嵌套结构

    参数:
        original: 原始字典 (将被修改)
        update_data: 包含更新数据的字典

    返回:
        更新后的原始字典 (原地修改)
    """
    for key, value in update_data.items():
        # 如果键存在于原始字典且两者都是字典类型，则递归更新
        if (key in original and
                isinstance(original[key], dict) and
                isinstance(value, dict)):
            recursive_dict_update(original[key], value)
        else:
            # 否则直接设置/覆盖值
            original[key] = value
    return original