
import torch

from torch.autograd.functional import _vmap,fwAD, _as_tuple, _grad_preprocess, _check_requires_grad, _construct_standard_basis_for, _autograd_grad, _grad_postprocess, _tuple_postprocess

def _jacfwd(func, inputs, strict=False, vectorize=False):
    if strict:
        raise RuntimeError('torch.autograd.functional.jacobian: `strict=True` '
                           'and `strategy="forward-mode"` are not supported together (yet). '
                           'Please either set `strict=False` or '
                           '`strategy="reverse-mode"`.')
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    output_info = []

    if vectorize:
        # See NOTE: [Computing jacobian with vmap and grad for multiple outputs]
        input_numels = tuple(input.numel() for input in inputs)

        # Step 1: Prepare tangents
        tangents = _construct_standard_basis_for(inputs, input_numels)

        # Step 2: Compute vmap over computation with dual tensors
        def jvp(tangents):
            with fwAD.dual_level():
                dual_inputs = tuple(
                    fwAD.make_dual(input, tangent.view_as(input)) for input, tangent in zip(inputs, tangents))
                _is_outputs_tuple, dual_outputs = _as_tuple(func(*dual_inputs), "outputs")
                output_info.append(_is_outputs_tuple)
                jv = []
                primal_outs = []
                for dual_out in dual_outputs:
                    primal, tangent = fwAD.unpack_dual(dual_out)
                    primal_outs.append(primal)
                    if tangent is not None:
                        jv.append(tangent)
                    else:
                        jv.append(torch.zeros_like(primal))
                output_info.append(primal_outs)
                return tuple(jv)

        outputs_before_split = _vmap(jvp)(tangents)
        is_outputs_tuple, outputs = output_info
        # Step 3: for each of the output tangents, split along dim 0
        jacobian_input_output = []
        for jac, output_i in zip(outputs_before_split, outputs):
            jacobian_output_i_output = []
            for jac, input_j in zip(jac.split(input_numels, dim=0), inputs):
                # We need to transpose the Jacobian because in forward AD, the
                # batch dimension represents that of the inputs
                jacobian_input_i_output_j = jac.permute(*range(1, jac.ndim), 0) \
                    .reshape(tuple([*output_i.shape, *input_j.shape]))  # noqa: C409

                jacobian_output_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_output_i_output)

        # Omit [Step 4] because everything is already transposed w/ forward AD
        return _tuple_postprocess(jacobian_input_output, (is_outputs_tuple, is_inputs_tuple))
    else:
        raise NotImplementedError("Computing Jacobian using forward-AD or forward-over-reverse Hessian is"
                                  "only implemented for `vectorize=True`.")



from torch.autograd.functional import _as_tuple, _grad_preprocess, _check_requires_grad, _construct_standard_basis_for, _autograd_grad, _grad_postprocess, _tuple_postprocess

from typing import List, Tuple
def jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy="reverse-mode"):
    r"""Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental.
            Please consider using
            `functorch's jacrev or jacfwd <https://github.com/pytorch/functorch#what-are-the-transforms>`_
            instead if you are looking for something less experimental and more performant.
            When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.

    Example:

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...   return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """
    assert strategy in ("forward-mode", "reverse-mode"), (
        'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
        'function has more outputs than inputs, "forward-mode" tends to be more performant. '
        'Otherwise, prefer to use "reverse-mode".')
    if strategy == "forward-mode":
        if create_graph:
            raise NotImplementedError('torch.autograd.functional.jacobian: `create_graph=True` '
                                      'and `strategy="forward-mode"` are not supported together (yet). '
                                      'Please either set `create_graph=False` or '
                                      '`strategy="reverse-mode"`.')
        return _jacfwd(func, inputs, strict, vectorize)

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs,
                                              "outputs of the user-provided function",
                                              "jacobian")
        _check_requires_grad(outputs, "outputs", strict=strict)

        if vectorize:
            if strict:
                raise RuntimeError('torch.autograd.functional.jacobian: `strict=True` '
                                   'and `vectorized=True` are not supported together. '
                                   'Please either set `strict=False` or '
                                   '`vectorize=False`.')
            # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
            #
            # Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
            # It turns out we can compute the jacobian of this function with a single
            # call to autograd.grad by using vmap over the correct grad_outputs.
            #
            # Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
            # into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
            #
            # To get the first row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
            # To get the 2nd row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
            # and so on.
            #
            # Using vmap, we can vectorize all 4 of these computations into one by
            # passing the standard basis for R^4 as the grad_output.
            # vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
            #
            # Now, how do we compute the jacobian *without stacking the output*?
            # We can just split the standard basis across the outputs. So to
            # compute the jacobian of f(x), we'd use
            # >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
            # The grad_outputs looks like the following:
            # ( torch.tensor([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]),
            #   torch.tensor([[0],
            #                 [0],
            #                 [0],
            #                 [1]]) )
            #
            # But we're not done yet!
            # >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
            # returns a Tensor of shape [4, 3]. We have to remember to split the
            # jacobian of shape [4, 3] into two:
            # - one of shape [3, 3] for the first output
            # - one of shape [   3] for the second output

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(_autograd_grad(flat_outputs, inputs, grad_output, create_graph=create_graph, is_grads_batched=True))
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx]).expand((sum(output_numels),) + inputs[el_idx].shape)
                return tuple(vj)

            jacobians_of_flat_output = vjp(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(jac.split(output_numels, dim=0), outputs):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            # Step 4: Right now, `jacobian` is a List[List[Tensor]].
            # The outer List corresponds to the number of inputs,
            # the inner List corresponds to the number of outputs.
            # We need to exchange the order of these and convert to tuples
            # before returning.
            jacobian_output_input = tuple(zip(*jacobian_input_output))

            jacobian_output_input = _grad_postprocess(jacobian_output_input, create_graph)
            return _tuple_postprocess(jacobian_output_input, (is_outputs_tuple, is_inputs_tuple))

        jacobian: Tuple[torch.Tensor, ...] = tuple()

        for i, out in enumerate(outputs):

            # mypy complains that expression and variable have different types due to the empty list
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in range(out.nelement()):
                vj = _autograd_grad((out.reshape(-1)[j],), inputs,
                                    retain_graph=True, create_graph=create_graph)

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = ("The jacobian of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode when create_graph=True.".format(i))
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = ("Output {} of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode.".format(i, el_idx))
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()  # type: ignore[operator]
                         + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)), )

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple)), tuple([output.detach() for output in outputs])


