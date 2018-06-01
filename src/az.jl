function POMDPs.solve(solver::AZSolver, mdp::Union{POMDP,MDP})
    S = state_type(mdp)
    A = action_type(mdp)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return AZPlanner(solver, mdp, Nullable{AZTree{S,A}}(), se, solver.next_action, solver.rng, true)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::AZPlanner)
    p.tree = Nullable()
end

"""
Construct an MCTS tree and choose the best action.
"""
POMDPs.action(p::AZPlanner, s) = first(action_info(p, s))

"""
Construct an MCTS tree and choose the best action. Also output some information.
"""
function POMDPToolbox.action_info(p::AZPlanner, s; tree_in_info=false)
    local a::action_type(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = state_type(p.mdp)
        A = action_type(p.mdp)
        if p.solver.keep_tree
            if isnull(p.tree)
                tree = AZTree{S,A}(p.solver.n_iterations)
                p.tree = Nullable(tree)
            else
                tree = get(p.tree)
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                v0 = estimate_value(p.solved_estimate, p.mdp, s, p.solver.depth)
                snode = insert_state_node!(tree, s, v0)
            end
        else
            tree = AZTree{S,A}(p.solver.n_iterations)
            p.tree = Nullable(tree)
            v0 = estimate_value(p.solved_estimate, p.mdp, s, p.solver.depth)
            snode = insert_state_node!(tree, s, v0, p.solver.check_repeat_state)
        end

        i = 0
        start_us = CPUtime_us()
        for i = 1:p.solver.n_iterations
            simulate(p, snode)
            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = i
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        all_actions = actions(p.mdp)   #Note, action nodes in the tree have the same order as here
        N = zeros(length(all_actions))
        N_sum = tree.total_n[snode]
        for (i,child) in enumerate(tree.children[snode])
            N[i] = tree.n[child]
        end
        action_distribution = (N./N_sum).^p.solver.tau
        if p.training_phase
            a = sample(all_actions,Weights(action_distribution))
        else
            a = all_actions[indmax(action_distribution)]
        end

        info[:action_distribution] = action_distribution

    catch ex
        a = convert(action_type(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTS.
"""
function simulate(az::AZPlanner, snode::Int)
    S = state_type(az.mdp)
    A = action_type(az.mdp)
    sol = az.solver
    tree = get(az.tree)
    s = tree.s_labels[snode]
    if isterminal(az.mdp, s)
        return 0.0
    end

    # action progressive widening
    if az.solver.enable_action_pw   #deprecated
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(az.next_action, az.mdp, s, AZStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, az.mdp, s, a)
                p0 = init_P(sol.init_P, az.mdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, az.mdp, s, a), p0,
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        allowed_actions = actions(az.mdp, s)   #This is handled a bit weird to make it compatible with existing structure of Multilane.jl
        all_actions = actions(az.mdp)
        if length(allowed_actions) == length(all_actions)
            allowed_actions_vec = ones(Float64, length(all_actions))
        else
            allowed_actions_vec = zeros(Float64, length(all_actions))
            for idx in collect(allowed_actions.acceptable)
                allowed_actions_vec[idx] = 1.0
            end
        end

        p0_vec = init_P(sol.init_P, az.mdp, s, allowed_actions_vec)
        if snode == 1 && az.training_phase   #Add Dirichlet noise to the root node
            distr = Dirichlet(length(allowed_actions),az.solver.noise_dirichlet)
            noise = rand(distr)   #ZZZ Warning, RNG does not work with Dirichlet, so does not help to give reproducible results
            j = 1
            for i in 1:length(p0_vec)
                if allowed_actions_vec[i] == 1.0
                    p0_vec[i] = (1-az.solver.noise_eps)*p0_vec[i] + az.solver.noise_eps*noise[j]
                    j += 1
                end
            end
        end

        for (i,a) in enumerate(all_actions)   #Loop through all actions, even the forbidden ones, but set their probabilities to 0
            n0 = init_N(sol.init_N, az.mdp, s, a)   #sol.initN set to 0
            p0 = p0_vec[i]
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, az.mdp, s, a), p0,   #sol.initQ set to 0
                                false)
            tree.total_n[snode] += n0
        end
    end

    best_UCB = -Inf
    sanode = 0
    tn = tree.total_n[snode]
    for child in shuffle(az.rng,tree.children[snode])   #Randomize in case of equal UCB values
        n = tree.n[child]
        q = tree.q[child]
        p = tree.p[child]
        c_puct = sol.exploration_constant # for clarity
        UCB = q + c_puct*p*sqrt(tn)/(1+n)
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end

    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
        sp, r = generate_sr(az.mdp, s, a, az.rng)

        spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0

        if spnode == 0 # there was not a state node for sp already in the tree
            v0 = estimate_value(az.solved_estimate, az.mdp, sp, az.solver.depth)
            spnode = insert_state_node!(tree, sp, v0, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end
        push!(tree.transitions[sanode], (spnode, r))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(az.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(az.mdp)*tree.v0[spnode]
    else
        q = r + discount(az.mdp)*simulate(az, spnode)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end
