import onnx
import json

class ONNXToNetworkStructure:
    def __init__(self, onnx_model_path):
        """
        初始化转换器

        Args:
            onnx_model_path: ONNX模型文件路径
        """
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.network_structure = []
        self.tensor_shapes = {}  # 存储中间张量的shape信息

    def _get_tensor_shape(self, tensor_name):
        """获取张量的shape"""
        if tensor_name in self.tensor_shapes:
            return self.tensor_shapes[tensor_name]

        # 从graph的value_info中查找
        for value_info in self.graph.value_info:
            if value_info.name == tensor_name:
                shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                self.tensor_shapes[tensor_name] = shape
                return shape

        # 从graph的input中查找
        for input_tensor in self.graph.input:
            if input_tensor.name == tensor_name:
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                self.tensor_shapes[tensor_name] = shape
                return shape

        return None

    def _infer_shapes(self):
        """推断所有张量的shape"""
        # 使用ONNX的shape inference
        try:
            inferred_model = onnx.shape_inference.infer_shapes(self.model)
            self.graph = inferred_model.graph

            # 提取所有tensor的shape信息
            for value_info in self.graph.value_info:
                if value_info.type.tensor_type.HasField('shape'):
                    shape = [dim.dim_value if dim.dim_value > 0 else -1
                             for dim in value_info.type.tensor_type.shape.dim]
                    self.tensor_shapes[value_info.name] = shape

            # 也提取input和output的shape
            for input_tensor in self.graph.input:
                if input_tensor.type.tensor_type.HasField('shape'):
                    shape = [dim.dim_value if dim.dim_value > 0 else -1
                             for dim in input_tensor.type.tensor_type.shape.dim]
                    self.tensor_shapes[input_tensor.name] = shape

            for output_tensor in self.graph.output:
                if output_tensor.type.tensor_type.HasField('shape'):
                    shape = [dim.dim_value if dim.dim_value > 0 else -1
                             for dim in output_tensor.type.tensor_type.shape.dim]
                    self.tensor_shapes[output_tensor.name] = shape

        except Exception as e:
            print(f"Shape inference warning: {e}")

    def _parse_conv_node(self, node):
        """解析Conv节点"""
        # 获取属性
        attrs = {attr.name: attr for attr in node.attribute}

        # 获取kernel_shape, strides, pads
        kernel_shape = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [3, 3]
        strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
        pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]

        # padding取上下左右的平均值（通常是相同的）
        padding = pads[0] if pads[0] == pads[1] == pads[2] == pads[3] else pads[0]

        # 获取输入输出shape
        input_name = node.input[0]
        output_name = node.output[0]

        input_shape = self._get_tensor_shape(input_name)
        output_shape = self._get_tensor_shape(output_name)

        if input_shape and output_shape:
            # NCHW格式: [batch, channels, height, width]
            in_channels = input_shape[1]
            in_h = input_shape[2]
            in_w = input_shape[3]

            out_channels = output_shape[1]
            out_h = output_shape[2]
            out_w = output_shape[3]

            conv_info = {
                "operator": "Conv",
                "in_W": in_w,
                "in_H": in_h,
                "in_channels": in_channels,
                "out_W": out_w,
                "out_H": out_h,
                "out_channels": out_channels,
                "kernel": kernel_shape,
                "stride": strides[0],
                "padding": padding
            }

            return conv_info

        return None

    def _parse_pool_node(self, node):
        """解析MaxPool节点"""
        attrs = {attr.name: attr for attr in node.attribute}

        # 获取kernel_shape, strides, pads
        kernel_shape = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [2, 2]
        strides = list(attrs['strides'].ints) if 'strides' in attrs else [2, 2]
        pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]

        padding = pads[0] if pads[0] == pads[1] == pads[2] == pads[3] else pads[0]

        # 获取输入输出shape
        input_name = node.input[0]
        output_name = node.output[0]

        input_shape = self._get_tensor_shape(input_name)
        output_shape = self._get_tensor_shape(output_name)

        if input_shape and output_shape:
            in_channels = input_shape[1]
            in_h = input_shape[2]
            in_w = input_shape[3]

            out_channels = output_shape[1]
            out_h = output_shape[2]
            out_w = output_shape[3]

            pool_info = {
                "operator": "Pool",
                "in_W": in_w,
                "in_H": in_h,
                "in_channels": in_channels,
                "out_W": out_w,
                "out_H": out_h,
                "out_channels": out_channels,
                "kernel": kernel_shape,
                "stride": strides[0],
                "padding": padding
            }

            return pool_info

        return None

    def _parse_fc_node(self, node):
        """解析全连接层（Gemm或MatMul + Add）"""
        # 获取输入输出shape
        input_name = node.input[0]
        output_name = node.output[0]

        input_shape = self._get_tensor_shape(input_name)
        output_shape = self._get_tensor_shape(output_name)

        # 检查前一层是否是FC
        is_prev_fc = False
        if len(self.network_structure) > 0:
            if self.network_structure[-1]["operator"] == "FC":
                is_prev_fc = True

        if input_shape and output_shape:
            # 对于FC层，输入通常是 [batch, features]
            in_features = input_shape[-1]
            out_features = output_shape[-1]

            fc_info = {
                "operator": "FC",
                "isPrevFC": is_prev_fc,
                "in_features": in_features,
                "out_features": out_features
            }

            return fc_info

        return None

    def convert(self):
        """执行转换"""
        # 首先推断所有shape
        self._infer_shapes()

        # 遍历所有节点
        for node in self.graph.node:
            layer_info = None

            # 处理Conv节点（包括ConvInteger）
            if node.op_type in ['Conv', 'ConvInteger']:
                layer_info = self._parse_conv_node(node)

            # 处理MaxPool节点
            elif node.op_type == 'MaxPool':
                layer_info = self._parse_pool_node(node)

            # 处理全连接层（Gemm或MatMulInteger）
            elif node.op_type in ['Gemm', 'MatMul', 'MatMulInteger']:
                layer_info = self._parse_fc_node(node)

            # 忽略其他类型的节点（Relu, BatchNorm, Quantize等）

            if layer_info:
                self.network_structure.append(layer_info)

        return self.network_structure

    def save_to_json(self, output_path):
        """保存为JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.network_structure, f, indent=4)
        print(f"Network structure saved to {output_path}")


def main():
    """主函数示例"""
    # 使用示例
    onnx_model_path = "Resnet640_cifar10_no_Normalize_int0810.onnx"
    output_json_path = "network_structure_output.json"

    # 创建转换器
    converter = ONNXToNetworkStructure(onnx_model_path)

    # 执行转换0
    network_structure = converter.convert()

    # 打印结果
    print("Extracted Network Structure:")
    print(json.dumps(network_structure, indent=4))

    # 保存到JSON文件
    converter.save_to_json(output_json_path)


if __name__ == "__main__":
    main()

