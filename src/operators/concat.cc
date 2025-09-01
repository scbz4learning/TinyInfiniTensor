#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================

    // ~~没必要 check 了~~
    // 还是要check， checkValid 不检查内部，只看最后的合不合理
    // for (size_t i = 1; i < inputs.size(); i++) {
    //     IT_ASSERT(rank == inputs[i]->getRank());
    //     for (size_t j = 0; j < rank; j++) {
    //         // 这里报错了，j 是 size_t 但是 dim 是 int
    //         // 为啥呢？负索引吧！加 rank 然后 assert 下
    //         if (j == dim)
    //             dims[j] += inputs[i]->getDims()[j];
    //         else
    //             IT_ASSERT(dims[j] == inputs[i]->getDims()[j]);
    //     }
    // }

    if (rank + dim < 0) {
        IT_ASSERT(1) << "dim is negative and out of bound";
    } else if (rank - dim < 0) {
        IT_ASSERT(1) << "dim is too large";
    }

    size_t realDim = dim < 0 ? rank + dim : dim;

    for (size_t i = 1; i < inputs.size(); i++) {
        IT_ASSERT(rank == inputs[i]->getRank());
        for (size_t j = 0; j < rank; j++) {
            if (j == realDim)
                dims[j] += inputs[i]->getDims()[j];
            else
                IT_ASSERT(dims[j] == inputs[i]->getDims()[j]);
        }
    }

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
