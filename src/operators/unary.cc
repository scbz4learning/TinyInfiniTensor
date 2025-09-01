#include "operators/unary.h"

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 clip 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
        // =================================== 作业 ===================================
        
        // `clip` is used for bond the element within max and min
        // so the shape is not augmented

        // 这个写法能通过test，但是做后面test的时候发现不太对
        // 因为这里语法是和onnx对齐的，clip接受的是 
        // [实际tensor, min tensor, max tensor]
        // 所以这样返回的是 [tensor dim, argument_min dim, argument_max dim]
        // 后两个肯定没用
        // 这里名字叫unary，所以显然是接受一个tensor的
        // vector<Shape> retDims;
        // for (auto tensor : inputs){
        //     retDims.push_back(tensor->getDims());
        // }
        // return retDims;

        return vector<Shape>{inputs[0]->getDims()};
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
        // REF_FILE: src/core/operator.cc
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================

        // 找不到ONNX的实现
        // 文档上说input 只有一个tensor
        // 另外有两个 attributes
        // 但是C++没有attributes
        // 只能通过test 来猜测inputs
        // auto op = g->addOp<CastObj>(i0, nullptr, CastType::Float2Float16);
        // CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type);
        // CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        // CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        //     : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
        // {
        //     IT_ASSERT(checkValid(graph));
        // }
        // 所以返回 private 成员 castType 就行

        // 应该是一堆tensor，type一样，所以用第一个代替？
        // 不是，CastObj 定义就一个
        // int numInputs() const override { return 1; }
        // int numOutputs() const override { return 1; }

        // castType 和 DataType 枚举类型不同！！！
        // return { DataType(static_cast<int>(castType)) };

        // castType 是 x to y 的形式，所以有专用函数吗
        // 有，下面的DataType CastObj::getOutputDataType() const
        return { getOutputDataType() };
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        // numOutputs 是 tensor 数量，不是每个tensor内的shape
        // return vector<Shape>{Shape(numOutputs())};
        return vector<Shape>{ inputs[0]->getDims() };
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
