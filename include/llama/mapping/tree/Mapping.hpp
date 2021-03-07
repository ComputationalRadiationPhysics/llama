// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Common.hpp"
#include "Functors.hpp"
#include "TreeFromDimensions.hpp"
#include "toString.hpp"

#include <type_traits>

namespace llama::mapping::tree
{
    namespace internal
    {
        template <typename Tree, typename TreeOperationList>
        struct MergeFunctors
        {
        };

        template <typename Tree, typename... Operations>
        struct MergeFunctors<Tree, Tuple<Operations...>>
        {
            boost::mp11::mp_first<Tuple<Operations...>> operation = {};
            using ResultTree = decltype(operation.basicToResult(Tree()));
            ResultTree treeAfterOp;
            MergeFunctors<ResultTree, boost::mp11::mp_drop_c<Tuple<Operations...>, 1>> next = {};

            MergeFunctors() = default;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctors(const Tree& tree, const Tuple<Operations...>& treeOperationList)
                : operation(treeOperationList.first)
                , treeAfterOp(operation.basicToResult(tree))
                , next(treeAfterOp, pop_front(treeOperationList))
            {
            }

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree& tree) const
            {
                if constexpr (sizeof...(Operations) > 1)
                    return next.basicToResult(treeAfterOp);
                else if constexpr (sizeof...(Operations) == 1)
                    return operation.basicToResult(tree);
                else
                    return tree;
            }

            template <typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& tree) const
            {
                if constexpr (sizeof...(Operations) >= 1)
                    return next.basicCoordToResultCoord(
                        operation.basicCoordToResultCoord(basicCoord, tree),
                        treeAfterOp);
                else
                    return basicCoord;
            }

            template <typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& tree) const
            {
                if constexpr (sizeof...(Operations) >= 1)
                    return next.resultCoordToBasicCoord(
                        operation.resultCoordToBasicCoord(resultCoord, tree),
                        operation.basicToResult(tree));
                else
                    return resultCoord;
            }
        };

        template <typename Tree>
        struct MergeFunctors<Tree, Tuple<>>
        {
            MergeFunctors() = default;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctors(const Tree&, const Tuple<>&)
            {
            }

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree& tree) const
            {
                return tree;
            }

            template <typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(TreeCoord const& basicCoord, Tree const& tree) const
                -> TreeCoord
            {
                return basicCoord;
            }

            template <typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(TreeCoord const& resultCoord, Tree const& tree) const
                -> TreeCoord
            {
                return resultCoord;
            }
        };

        template <typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t;

        template <typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t;

        template <typename... Children, std::size_t... Is, typename Count>
        LLAMA_FN_HOST_ACC_INLINE auto getChildrenBlobSize(
            const Tuple<Children...>& childs,
            std::index_sequence<Is...> ii,
            const Count& count) -> std::size_t
        {
            return count * (getTreeBlobSize(get<Is>(childs)) + ...);
        }

        template <typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t
        {
            constexpr std::size_t childCount = boost::mp11::mp_size<std::decay_t<decltype(node.childs)>>::value;
            return getChildrenBlobSize(node.childs, std::make_index_sequence<childCount>{}, LLAMA_COPY(node.count));
        }

        template <typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t
        {
            return leaf.count * sizeof(Type);
        }

        template <typename Childs, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Childs& childs, const CountType& count) -> std::size_t
        {
            return getTreeBlobSize(Node<NoName, Childs, CountType>{count, childs});
        }

        template <std::size_t MaxPos, typename Identifier, typename Type, typename CountType, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto sumChildrenSmallerThan(
            const Node<Identifier, Type, CountType>& node,
            std::index_sequence<Is...>) -> std::size_t
        {
            return ((getTreeBlobSize(get<Is>(node.childs)) * (Is < MaxPos)) + ...);
        }

        template <typename Tree, typename... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobByte(const Tree& tree, const Tuple<Coords...>& treeCoord)
            -> std::size_t
        {
            const auto firstArrayIndex = treeCoord.first.arrayIndex;
            if constexpr (sizeof...(Coords) > 1)
            {
                constexpr auto firstChildIndex = decltype(treeCoord.first.childIndex)::value;
                return getTreeBlobSize(tree.childs, firstArrayIndex)
                    + sumChildrenSmallerThan<firstChildIndex>(
                           tree,
                           std::make_index_sequence<std::tuple_size_v<typename Tree::ChildrenTuple>>{})
                    + getTreeBlobByte(get<firstChildIndex>(tree.childs), treeCoord.rest);
            }
            else
                return sizeof(typename Tree::Type) * firstArrayIndex;
        }
    } // namespace internal

    /// An experimental attempt to provide a general purpose description of a mapping. \ref ArrayDims and record
    /// dimension are represented by a compile time tree data structure. This tree is mapped into memory by means of a
    /// breadth-first tree traversal. By specifying additional tree operations, the tree can be modified at compile time
    /// before being mapped to memory.
    template <typename T_ArrayDims, typename T_RecordDim, typename TreeOperationList>
    struct Mapping
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        using BasicTree = TreeFromDimensions<ArrayDims, RecordDim>;
        // TODO, support more than one blob
        static constexpr std::size_t blobCount = 1;

        using MergedFunctors = internal::MergeFunctors<BasicTree, TreeOperationList>;

        ArrayDims arrayDimsSize = {};
        BasicTree basicTree;
        MergedFunctors mergedFunctors;

        using ResultTree = decltype(mergedFunctors.basicToResult(basicTree));
        ResultTree resultTree;

        Mapping() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Mapping(ArrayDims size, TreeOperationList treeOperationList, RecordDim = {})
            : arrayDimsSize(size)
            , basicTree(createTree<RecordDim>(size))
            , mergedFunctors(basicTree, treeOperationList)
            , resultTree(mergedFunctors.basicToResult(basicTree))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto blobSize(std::size_t const) const -> std::size_t
        {
            return internal::getTreeBlobSize(resultTree);
        }

        template <std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord, Array<std::size_t, N> dynamicArrayExtents = {})
            const -> NrAndOffset
        {
            auto const basicTreeCoord = createTreeCoord<RecordCoord<RecordCoords...>>(coord);
            auto const resultTreeCoord = mergedFunctors.basicCoordToResultCoord(basicTreeCoord, basicTree);
            const auto offset = internal::getTreeBlobByte(resultTree, resultTreeCoord);
            return {0, offset};
        }
    };
} // namespace llama::mapping::tree
