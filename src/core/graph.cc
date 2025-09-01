#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    // 没 lock
    // void GraphObj::optimize()
    // {
    //     // =================================== 作业 ===================================
    //     // TODO: 设计一个算法来实现指定的图优化规则
    //     // 图优化规则如下：
    //     // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    //     // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    //     // =================================== 作业 ===================================

    //     // 看 test 好像 只做 transpose 优化
    //     IT_ASSERT(topo_sort() == true);

    //     // 如果第一个是T，最后一个op肯定不需要优化
    //     for (auto itOp = ops.begin(); itOp < ops.end()-1;) {
    //         if ((*itOp)->getOpType() == OpType::Transpose) {
    //             // T -> T
    //             // T has 1 output and input only
    //             if ((*itOp)->getSuccessors()[0]->getOpType() == OpType::Transpose) {
    //                 // 只有 replaceInput 接口
    //                 // 说明改后不改前
    //                 // // 前算子改 outputs
    //                 // (*itOp)->getPredecessors()[0]->;

    //                 // 改前张量targets
    //                 (*itOp)->getInputs()[0]->removeTarget((*itOp));
    //                 (*itOp)->getInputs()[0]->addTarget(
    //                     (*itOp)->getSuccessors()[0]->getSuccessors()[0]
    //                 );

    //                 // 改后后算子input
    //                 (*itOp)->getSuccessors()[0]->getSuccessors()[0]->replaceInput(
    //                     (*itOp)->getSuccessors()[0]->getOutput(0), 
    //                     (*itOp)->getInputs()[0]
    //                 );

    //                 // remove 本算子output 张量 和 后算子output张量
    //                 removeTensor((*itOp)->getOutput(0));
    //                 removeTensor((*itOp)->getSuccessors()[0]->getOutput(0));

    //                 // remove 本算子 和 后算子
    //                 itOp = ops.erase(itOp);
    //                 itOp = ops.erase(itOp);
    //             }
    //             else if ((*itOp)->getSuccessors()[0]->getOpType() == OpType::MatMul) {
    //                 // 改前张量targets
    //                 (*itOp)->getInputs()[0]->removeTarget((*itOp));
    //                 (*itOp)->getInputs()[0]->addTarget(
    //                     (*itOp)->getSuccessors()[0]
    //                 );

    //                 // 改后算子input
    //                 auto succ = (*itOp)->getSuccessors()[0];
    //                 auto matmul = std::dynamic_pointer_cast<MatmulObj>(succ);
    //                 auto transposeOutputTensor = matmul->getInputs();
    //                 if (transposeOutputTensor[0] == (*itOp)->getOutput(0)) {
    //                     matmul->setTransA(!matmul->getTransA());
    //                 } else if (transposeOutputTensor[1] == (*itOp)->getOutput(0)) {
    //                     matmul->setTransB(!matmul->getTransB());
    //                 }

    //                 // remove 本算子output 张量
    //                 removeTensor((*itOp)->getOutput(0));

    //                 // remove 本算子
    //                 itOp = ops.erase(itOp);
    //             }
    //             else {
    //                 itOp++;
    //             }
    //         } else {
    //             itOp++;
    //         }
    //     }        
    // }

    // Tensor 和 Op 是不一样的，tensor中 source 和 targets 是 WRef，要lock
    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        // 看 test 好像 只做 transpose 优化
        IT_ASSERT(topo_sort() == true);

        // 如果第一个是T，最后一个op肯定不需要优化
        for (auto itOp = ops.begin(); itOp < ops.end()-1;) {
            if ((*itOp)->getOpType() == OpType::Transpose) {
                auto op_output = (*itOp)->getOutput();
                auto op_input = (*itOp)->getInputs()[0];

                // T -> T
                // T has 1 output and input only
                if ((*itOp)->getSuccessors()[0]->getOpType() == OpType::Transpose) {
                    auto succ = (*itOp)->getSuccessors()[0];
                    auto succ_succ = succ->getSuccessors()[0];

                    // 只有 replaceInput 接口
                    // 说明改后不改前
                    // // 前算子改 outputs
                    // (*itOp)->getPredecessors()[0]->;

                    // 改前张量targets
                    (*itOp)->getInputs()[0]->removeTarget((*itOp));
                    (*itOp)->getInputs()[0]->addTarget(
                        (*itOp)->getSuccessors()[0]->getSuccessors()[0]
                    );

                    // 改后后算子input
                    (*itOp)->getSuccessors()[0]->getSuccessors()[0]->replaceInput(
                        (*itOp)->getSuccessors()[0]->getOutput(0),
                        (*itOp)->getInputs()[0]
                    );

                    // remove 本算子output 张量 和 后算子output张量
                    removeTensor((*itOp)->getOutput(0));
                    removeTensor((*itOp)->getSuccessors()[0]->getOutput(0));

                    // remove 本算子 和 后算子
                    // 先断开关系再删除算子
                    for (auto pred : (*itOp)->getPredecessors()) {
                        pred->removeSuccessors(*itOp);
                    }
                    for (auto succ_pred : succ->getSuccessors()) {
                        succ_pred->removePredecessors(succ);
                    }

                    itOp = ops.erase(itOp);
                    itOp = ops.erase(itOp);
                }
                else if ((*itOp)->getSuccessors()[0]->getOpType() == OpType::MatMul) {
                    auto succ = (*itOp)->getSuccessors()[0]; // MatMul
                    auto matmul = std::dynamic_pointer_cast<MatmulObj>(succ);

                    auto trans_input = (*itOp)->getInputs()[0];
                    auto trans_output = (*itOp)->getOutput();

                    // 改后算子input
                    if (matmul->getInputs()[0] == trans_output) {
                        matmul->setTransA(!matmul->getTransA());
                        matmul->replaceInput(trans_output, trans_input);
                    } else if (matmul->getInputs()[1] == trans_output) {
                        matmul->setTransB(!matmul->getTransB());
                        matmul->replaceInput(trans_output, trans_input);
                    }

                    // 改后算子input
                    trans_input->removeTarget(*itOp);
                    trans_input->addTarget(matmul);

                    // remove 本算子output 张量
                    removeTensor((*itOp)->getOutput(0));

                    // remove 本算子
                    // 断开算子关系再删除
                    for (auto pred : (*itOp)->getPredecessors()) {
                        pred->removeSuccessors(*itOp);
                    }
                    for (auto succ_succ : (*itOp)->getSuccessors()) {
                        succ_succ->removePredecessors(*itOp);
                    }
                    itOp = ops.erase(itOp);
                }
                else {
                    itOp++;
                }
            } else {
                itOp++;
            }
        }        
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        // 这里逻辑还是没顺清楚
        // 静态分配之后，哪里free了？
        //      - 析构的时候！Ref为0，自动析构，析构销毁 X 析构函数中没有
        // 为什么先后绑定带来差异？

        // 这里有点像制作纸带：
        //      应该按照这个逻辑来分配内存哦，后面计算的时候就用我给的offset去申请内存
        // 所以理论上是可以动态分配的。
        // 因为中间变量一定再输出的时候产生，最后一次输入的时候销毁
        // 用hash map计数，遍历op，每个op输出的tensor在hash map 不存在就alloc
        //      存在就++。输入的tensor -- ，如果归零就free
        // 原版的Infinitensor也是这样做的
        // 然而这里有个问题，最初的输入没法alloc
        // 因为原版有isWeight()，这里没有。所以只能静态分配不复用
        // 不过allocator也不是没用，因为一个allocator可以管多个图，多个图之间还是可以复用的
        
        // // 用来记录每个 tensor 的起始偏移
        // std::unordered_map<Tensor, size_t> tensorToOffset;

        // // 遍历所有 tensor，静态分配内存
        // for (auto &tensor : tensors) {
        //     // 分配内存
        //     size_t offset = allocator.alloc(tensor->getBytes());
        //     tensorToOffset[tensor] = offset;

        //     // 立即绑定 Blob
        //     auto blob = make_ref<BlobObj>(runtime, allocator.getPtr() + offset);
        //     tensor->setDataBlob(blob);
        // }


        // 立即绑定 Blob 不能通过test，因为同时分配了同一段内存吗？但是之后再申请也会这样不是吗？
        // 有那个函数调用free了吗？
        // std::unordered_map<TensorObj*, size_t> tensorToOffset;
        //
        // // 遍历所有 tensor，静态分配内存
        // for (auto &tensor : tensors) {
        //     // 分配内存
        //     size_t offset = allocator.alloc(tensor->getBytes());
        //     tensorToOffset[tensor.get()] = offset;  // 注意这里用裸指针作为 key
        //
        //     // 立即绑定 Blob
        //     auto blob = make_ref<BlobObj>(runtime, allocator.getPtr() + offset);
        //     tensor->setDataBlob(blob);
        // }


        // 如果不能遍历的时候立即绑定，这里好像用 vector 更好
        std::unordered_map<Tensor, size_t> tensorToOffset;

        for (auto &tensor : tensors) {
            size_t offset = allocator.alloc(tensor->getBytes());
            tensorToOffset[tensor] = offset;
        }

        auto ptr = allocator.getPtr();

        for (auto &kv : tensorToOffset) {
            auto &tensor = kv.first;
            size_t offset = kv.second;
            auto blob = make_ref<BlobObj>(runtime, ptr + offset);
            tensor->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini