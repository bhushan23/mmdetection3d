from mmdet3d.apis import init_model

import warnings
import numpy as np
import torch
import pickle
import pytest

CONFIG = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CHECKPOINT = "pretrained/bevfusion_converted.pth"

class CamEncoder(torch.nn.Module):
    def __init__(self, model):
        super(CamEncoder, self).__init__()
        self.model = model

    def forward(self, x, points, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix):
        return self.model.extract_img_feat(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas=None)

class FusionLayer(torch.nn.Module):
    def __init__(self, model):
        super(FusionLayer, self).__init__()
        self.model = model

    def forward(self, bev_pool_output, pts_feature):
        """
        bev_pool_output:
        """
        img_feature = self.model.view_transform.downsample(bev_pool_output)
        return self.model.fusion_layer([img_feature, pts_feature])

class SecondBackbone(torch.nn.Module):
    def __init__(self, model):
        super(SecondBackbone, self).__init__()
        self.model = model

    def forward(self, fusion_layer_output):
        return self.model.pts_backbone(fusion_layer_output)

class SecondFPN(torch.nn.Module):
    def __init__(self, model):
        super(SecondFPN, self).__init__()
        self.model = model

    def forward(self, pts_backbone_output1, pts_backbone_output2):
        return self.model.pts_neck([pts_backbone_output1, pts_backbone_output2])

class BBoxHead(torch.nn.Module):
    def __init__(self, model):
        super(BBoxHead, self).__init__()
        self.model = model

    def forward(self, pts_neck_output):
        res = self.model.bbox_head.forward_export(pts_neck_output)
        return self.model.bbox_head.predict_by_feat_export(res[0])

class FusionBBox(torch.nn.Module):
    """
    Runs the followings:
    - downsample (last layer of cam encoder, after the bev_pool)
    - fusion_layer (input1: output from extract_pts_feat, input2: downsample result)
    - bbox detection layers
    """
    def __init__(self, model):
        super(FusionBBox, self).__init__()
        self.model = model

    def forward(self, bev_pool_output, pts_feature):
        """
        bev_pool_output:
        """
        img_feature = self.model.view_transform.downsample(bev_pool_output)
        x = self.model.fusion_layer([img_feature, pts_feature])
        x = self.model.pts_backbone(x)
        import pdb
        pdb.set_trace()
        # pts_neck: mmdet3d.models.necks.second_fpn.SECONDFPN
        x = self.model.pts_neck(x)
        res = self.model.bbox_head.forward_export(x)
        #return tuple(res[0][0].values())

        # The rest can export to onnx but gets an error during onnx -> tf:
        #>       assert 3 <= trank <= 5 and orank >= trank - 1
        #E       AssertionError
        # ../../tetracode/src/www/python/tetrai/compilers/onnx2tf/utils/common_functions.py:96: AssertionErrorCannot 
        res = self.model.bbox_head.predict_by_feat_export(res[0])
        return res

#class BBoxHead(torch.nn.Module):
#    def __init__(self, model):
#        super(BBoxHead, self).__init__()
#        self.model = model
#
#    def forward(self, bbox_input):
#        res = self.model.bbox_head([bbox_input])
#
#        # res[0][0] is a dict. Nothing else in res
#        # {'center': torch.Size([1, 2, 200]), 'height': torch.Size([1, 1, 200]), 'dim': torch.Size([1, 3, 200]), 'rot': torch.Size([1, 2, 200]), 'vel': torch.Size([1, 2, 200]), 'heatmap': torch.Size([1, 10, 200]), 'query_heatmap_score': torch.Size([1, 10, 200]), 'dense_heatmap': torch.Size([1, 10, 180, 180])}
#
#        #return x
#        #return res[0][0]
#        return tuple(res[0][0].values())

class LidarEncoder(torch.nn.Module):
    def __init__(self, model):
        super(LidarEncoder, self).__init__()
        self.model = model

    def forward(self, points):
        return self.model.extract_pts_feat(points)

# class BEVFusionSparseEncoder(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, feats, coords, batch_size=1):
#         """
#         bev_pool_output:
#         """
#         # return self.model.pts_middle_encoder(feats, coords, batch_size)
#         return self.model(feats, coords, batch_size)


@pytest.fixture(scope="module")
def bev_model():
    return init_model(CONFIG, CHECKPOINT, device="cpu").eval()

def test_convert_cam_encoder(bev_model):
    with open('sample_input.pkl', 'rb') as f:
        sample_input = pickle.load(f)
    wrapper = CamEncoder(bev_model)
    input_example = (
            sample_input["x"],
            sample_input["points"],
            sample_input["lidar2image"],
            sample_input["camera_intrinsics"],
            sample_input["camera2lidar"],
            sample_input["img_aug_matrix"],
            sample_input["lidar_aug_matrix"],
            )
    torch.onnx.export(wrapper, input_example,
                      "export/img_extractor.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, "export/img_extractor.jit")

def test_convert_lidar_encoder(bev_model):
    with open('sample_input.pkl', 'rb') as f:
        sample_input = pickle.load(f)
    wrapper = LidarEncoder(bev_model)
    input_example = (sample_input["points"],)
    torch.onnx.export(wrapper, input_example,
                      "export/lidar_encoder.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, "export/lidar_encoder.jit")

def test_convert_fusion_layer(bev_model):
    export_name = "fusion_layer"
    bev_pool_data = torch.load('bhushan_data/cam_encoder_BaseDepthTransform_bev_pool_input_output.pt')
    
    fusion_data = torch.load('bhushan_data/fusion_layer_input_output.pt')
    assert len(fusion_data["features"]) == 2  # [img_feature, pts_feature]
    pts_feature = fusion_data["features"][1]
    wrapper = FusionLayer(bev_model)
    input_example = (bev_pool_data["bev_pool_output"].float(), pts_feature.float(),)
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit")

def test_convert_second_backbone(bev_model):
    export_name = "second_backbone"
    fusion_data = torch.load('bhushan_data/fusion_layer_input_output.pt')
    wrapper = SecondBackbone(bev_model)
    input_example = (fusion_data["feasuion_layer_output"].float(),)
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit")

def test_convert_second_fpn(bev_model):
    export_name = "second_fpn"
    fusion_data = torch.load('bhushan_data/fusion_layer_input_output.pt')
    wrapper = SecondFPN(bev_model)
    second_backbone_output = fusion_data["pts_backbone_output"]
    input_example = (second_backbone_output[0].float(), second_backbone_output[1].float())
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit")

def test_convert_bbox_head(bev_model):
    export_name = "bbox_head"
    fusion_data = torch.load('bhushan_data/fusion_layer_input_output.pt')
    wrapper = BBoxHead(bev_model)
    input_example = (fusion_data["pts_neck_output"].float(),)
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit")

def test_convert_fusion_bbox(bev_model):
    export_name = "fusion_bbox"
    bev_pool_data = torch.load('bhushan_data/cam_encoder_BaseDepthTransform_bev_pool_input_output.pt')
    
    fusion_data = torch.load('bhushan_data/fusion_layer_input_output.pt')
    assert len(fusion_data["features"]) == 2  # [img_feature, pts_feature]
    pts_feature = fusion_data["features"][1]
    wrapper = FusionBBox(bev_model)
    input_example = (bev_pool_data["bev_pool_output"].float(), pts_feature.float(),)
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit")

def test_convert_bbox_head(bev_model):
    export_name = "bbox_head"
    bbox_head_input = torch.load('bhushan_data/bbox_head_input.pt')
    wrapper = BBoxHead(bev_model)
    input_example = (bbox_head_input.float(),)
    torch.onnx.export(wrapper, input_example,
                      f"export/{export_name}.onnx")
    traced_model = torch.jit.trace(wrapper, input_example)
    torch.jit.save(traced_model, f"export/{export_name}.jit.pt")


class BEVFusionSparseEncoder_ConvInput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats, coords, batch_size=1):
        """
        bev_pool_output:
        """
        # return self.model.pts_middle_encoder(feats, coords, batch_size)
        return self.model.forward_conv_in(feats, coords, batch_size)

class BEVFusionSparseEncoder_EncoderLayers(torch.nn.Module):
    def __init__(self, model, start=0, end=100):
        super().__init__()
        self.model = model
        self.start = start
        self.end = end

    def forward(self, feats):
        """
        bev_pool_output:
        """
        # return self.model.pts_middle_encoder(feats, coords, batch_size)
        return self.model.forward_encoder_layers(feats, self.start, self.end)
    
class BEVFusionSparseEncoder_ConvOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats):
        """
        bev_pool_output:
        """
        # return self.model.pts_middle_encoder(feats, coords, batch_size)
        return self.model.forward_conv_outs(feats)

    

def test_convert_sparse_encoder_conv_in(bev_model):
    with torch.no_grad():
        export_name = "bev_sparse_encoder_conv_in"
        data = torch.load('bhushan_data/bev_fusion_sparse_encoder_conv_in.pt')
        wrapper = BEVFusionSparseEncoder_ConvInput(bev_model.pts_middle_encoder).to('cpu') #.to('cpu')
        wrapper.eval()
        input_example = (data['voxel_features'], data['coors']) #, data['batch_size'])
        print(input_example[0].shape, input_example[1].shape)
        torch.onnx.export(wrapper, input_example,
                          f"export/{export_name}.onnx", opset_version=11)
        traced_model = torch.jit.trace(wrapper, input_example, check_trace=False)
        torch.jit.save(traced_model, f"export/{export_name}.jit.pt")


def test_convert_sparse_encoder_encoder_layers(bev_model):
    with torch.no_grad():
        export_name = "bev_sparse_encoder_encoder_layers"
        data = torch.load('bhushan_data/bev_fusion_sparse_encoder_encoder_layers.pt')
        wrapper = BEVFusionSparseEncoder_EncoderLayers(bev_model.pts_middle_encoder, 0, 1).to('cpu') #.to('cpu')
        wrapper.eval()
        input_example = (data['x'][:, :600, :600, :]) #, data['batch_size'])
        # print(input_example[0].shape)
        torch.onnx.export(wrapper, input_example,
                          f"export/{export_name}.onnx", opset_version=11)
        # traced_model = torch.jit.trace(wrapper, input_example, check_trace=False)
        # torch.jit.save(traced_model, f"export/{export_name}.jit.pt")

def test_convert_sparse_encoder_conv_out(bev_model):
    with torch.no_grad():
        export_name = "bev_sparse_encoder_conv_out"
        data = torch.load('bhushan_data/bev_fusion_sparse_encoder_conv_out.pt')
        wrapper = BEVFusionSparseEncoder_ConvOutput(bev_model.pts_middle_encoder).to('cpu') #.to('cpu')
        wrapper.eval()
        input_example = (data['voxel_features']) #, data['batch_size'])
        # print(input_example[0].shape, input_example[1].shape)
        torch.onnx.export(wrapper, input_example,
                          f"export/{export_name}.onnx", opset_version=11)
        traced_model = torch.jit.trace(wrapper, input_example, check_trace=False)
        torch.jit.save(traced_model, f"export/{export_name}.jit.pt")

def test_convert_sparse(bev_model):
    import os
    import psutil
    from concurrent import futures

    #JIT trace test
    import torch



    with torch.no_grad():
        # Create a simple resnet

        # Get process id
        process = psutil.Process(os.getpid())

        # Repeat tracing
        for i in range(0, 1000):
            with futures.ProcessPoolExecutor() as executor:
                future = executor.submit(test_convert_sparse_encoder)
                futures.wait([future])
            print("Iter: {} = {}".format(i, process.memory_full_info()))