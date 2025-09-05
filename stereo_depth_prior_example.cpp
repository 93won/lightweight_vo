// YGZ-SLAM의 Stereo Depth Prior 적용 방법을 참고한 예제

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>

// Inverse Depth Prior Edge (YGZ 방식)
class EdgeInverseDepthPrior : public g2o::BaseUnaryEdge<1, double, VertexInverseDepth>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInverseDepthPrior() = default;

    void computeError() override
    {
        const VertexInverseDepth* v = static_cast<const VertexInverseDepth*>(_vertices[0]);
        _error[0] = v->estimate() - _measurement;  // prior와의 차이
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi[0] = 1.0;  // d(error)/d(inv_depth) = 1
    }

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }
};

// YGZ 방식의 Stereo Depth Prior 적용
void addStereoDepthPrior(g2o::SparseOptimizer& optimizer,
                        const Feature& feat,
                        VertexInverseDepth* v_inv_depth,
                        double baseline_fx)  // bf = baseline * fx
{
    if (feat.stereo_inv_depth > 0) {  // 유효한 stereo depth가 있을 때만
        
        // Prior edge 생성
        EdgeInverseDepthPrior* e_prior = new EdgeInverseDepthPrior();
        e_prior->setVertex(0, v_inv_depth);
        e_prior->setMeasurement(feat.stereo_inv_depth);
        
        // YGZ 방식의 Information Matrix 계산
        const double inv_sigma2 = 1.0 / (feat.pyramid_level_sigma * feat.pyramid_level_sigma);
        const double info_weight = inv_sigma2 * 0.5 * baseline_fx * baseline_fx;
        
        Eigen::Matrix<double, 1, 1> information;
        information(0, 0) = info_weight;
        e_prior->setInformation(information);
        
        // Robust kernel 적용 (선택사항)
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(sqrt(5.991));  // Chi-square 95% threshold
        e_prior->setRobustKernel(rk);
        
        optimizer.addEdge(e_prior);
    }
}

// Bundle Adjustment에서 사용 예제
void bundleAdjustmentWithStereoPrior()
{
    g2o::SparseOptimizer optimizer;
    // ... optimizer setup ...
    
    for (auto& mappoint : mappoints) {
        // Map point vertex 추가
        VertexInverseDepth* v_inv_depth = new VertexInverseDepth();
        v_inv_depth->setEstimate(mappoint.inv_depth);
        v_inv_depth->setId(mappoint.id);
        optimizer.addVertex(v_inv_depth);
        
        // 해당 map point를 관찰하는 모든 feature에 대해
        for (auto& observation : mappoint.observations) {
            Frame* frame = observation.frame;
            Feature* feat = observation.feature;
            
            // 1. Reprojection edge 추가 (기본)
            // addReprojectionEdge(optimizer, frame, feat, v_inv_depth);
            
            // 2. Stereo depth가 있으면 prior edge 추가 (YGZ 방식)
            if (feat->has_stereo_depth) {
                addStereoDepthPrior(optimizer, *feat, v_inv_depth, frame->camera->bf);
            }
        }
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
}
