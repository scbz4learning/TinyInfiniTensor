#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 看 test，input 会传 A, B, C (output, 可以nullptr), transA (bool = false), transB (bool = false) 
        // IT_ASSERT(inputs.size() != 2) << "There should be 2 matrices";

        Shape A = inputs[0]->getDims();
        Shape B = inputs[1]->getDims();

        IT_ASSERT(A.size() >=2 && B.size() >= 2);

        if (getTransA()) {
            std::swap(A[A.size() - 1], A[A.size() - 2]);
        }
        if (getTransB()) {
            std::swap(B[B.size() - 1], B[B.size() - 2]);
        }

        Shape retShape(std::max(A.size(), B.size()));

        IT_ASSERT(A[A.size()-1] == B[B.size()-2]) \
            << "The mismatched n in (.., m n) (.., n,k)";
        retShape[retShape.size() - 1] = B[B.size() - 1];
        retShape[retShape.size() - 2] = A[A.size() - 2];

        // broadcast
        for (size_t i = 2; i < retShape.size(); i++) {
            size_t dimA = i < A.size() ? A[A.size() - 1 - i] : 1;
            size_t dimB = i < B.size() ? B[B.size() - 1 - i] : 1;
            IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1) 
                << "mismatch dim in broadcasting";
            retShape[retShape.size() - 1 - i] = dimA != 1 ? dimA : dimB; 
        }

        return vector<Shape>{ retShape };
    }

} // namespace infini