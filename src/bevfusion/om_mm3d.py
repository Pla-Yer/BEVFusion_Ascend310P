import torch
import numpy as np
from mmdet3d.registry import MODELS
from mmdet3d.models import Base3DDetector
# 导入你现有的 Net 和 ACL 初始化工具
from bevfusion_net import Net, init_acl 

@MODELS.register_module()
class BEVFusionOM(Base3DDetector):
    def __init__(self, 
                 om_model_path, 
                 bbox_head,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # 1. 初始化昇腾 ACL 环境和加载模型
        init_acl(0)
        self.net = Net(om_model_path)
        
        # 2. 构造标准的 TransFusionHead 用于后处理
        # 这样我们可以直接复用 head.predict_by_feat 方法
        self.bbox_head = MODELS.build(bbox_head)

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        """
        这里的输入是经过 mmdet3d DataPreprocessor 处理后的 Tensor。
        我们需要将其转换为 numpy 喂给 .om 模型。
        """
        # 根据你 export_bevfusion_full0306.py 的导出逻辑，需要三个输入
        # voxels: [V, M, C], num_points: [V], coors: [V, 4]
        voxels = batch_inputs_dict['voxels']['voxels'].cpu().numpy()
        num_points = batch_inputs_dict['voxels']['num_points'].cpu().numpy().astype(np.int32)
        coords = batch_inputs_dict['voxels']['coors'].cpu().numpy().astype(np.int32)

        # 调用你 bevfusion_net.py 中的 forward
        # 它内部已经处理了 acl.rtMemcpy 和 model_execute
        om_outputs = self.net.forward([voxels, num_points, coords])

        # 将 OM 的输出转回 torch.Tensor 并放到 GPU 上，给 Head 做后处理
        # 假设输出顺序是：heatmap, center, height, dim, rot, vel
        # 需要确保这些 Tensor 的 Shape 和 TransFusionHead.predict_by_feat 期待的一致
        torch_outputs = [torch.from_numpy(res).cuda() for res in om_outputs]
        
        # TransFusionHead 期待的是一个 List，其中包含最后的 Feature Map 输出
        # 如果你的 OM 导出的是多层结果，这里需要按照 head 的逻辑组装
        return torch_outputs

    def _forward(self, batch_inputs_dict, batch_data_samples):
        """覆盖基类方法，直接走 extract_feat"""
        return self.extract_feat(batch_inputs_dict, None)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """
        这是被 mm3d test.py 调用的核心入口
        """
        # 1. 运行 OM 推理得到原始 Tensor
        preds_dict = self.extract_feat(batch_inputs_dict, None)
        
        # 2. 这里的 preds_dict 需要包装成 TransFusionHead 能识别的格式
        # 具体的 key 名取决于你的 TransFusionHead.predict_by_feat 实现
        # 通常 TransFusionHead 需要一个包含 center, height 等 key 的字典列表
        
        # 如果你的 OM 导出的就是单个结果，可以模拟一个 layer_res
        res_layer = {
            'heatmap': preds_dict[0],
            'center': preds_dict[1],
            'height': preds_dict[2],
            'dim': preds_dict[3],
            'rot': preds_dict[4],
            'vel': preds_dict[5]
        }
        
        # 3. 调用原生的后处理逻辑（NMS, 阈值过滤, 坐标转换）
        # 注意：这里需要传入 batch_data_samples 获取元信息（如 lidar2global 变换）
        results_list = self.bbox_head.predict_by_feat([res_layer], batch_data_samples)
        
        # 4. 封装回 mm3d 的 DataSample 格式
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def loss(self, **kwargs):
        raise NotImplementedError("OM model is only for inference.")