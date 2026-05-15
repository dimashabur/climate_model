module climate_model_v2

using DifferentialEquations, Plots, CSV, DataFrames

# ── Fixed parameters ─────────────────────────────────────────────────────────
const α  = 0.33     # net climate sensitivity coefficient
const λ  = 1.2      # temperature damping coefficient
const β  = 5.35     # CO₂ radiative forcing coefficient
const C★ = 278.0    # pre-industrial reference carbon level (ppm)
const γ  = 0.015    # carbon absorption rate (yr⁻¹)
const ϕ  = 11.3     # carbon emission factor
const s  = 0.25     # savings rate
const A₀ = 0.65     # baseline TFP (calibrated for ~6 % yr⁻¹ initial growth; K equilibrates near 2)
const θ  = 0.3      # capital output elasticity
const δ  = 0.1      # capital depreciation rate (yr⁻¹)
const η  = 0.00236  # temperature damage coefficient
const κ  = 0.05     # mitigation cost coefficient
const n  = 2        # Hill coefficient
const Tc = 2.0      # critical temperature threshold (°C)

# ── Initial conditions and time spans ────────────────────────────────────────
const u0        = [1.93, 429.0, 1.0, 0.05]  # [T(0), C(0), K(0), M(0)] — T(0) on temperature nullcline
const tspan     = (0.0, 100.0)              # 100-year horizon
const tspan_long = (0.0, 500.0)             # 500-year horizon

# ── ODE systems ──────────────────────────────────────────────────────────────

function ode_v1!(du, u, p, t)
    T, C, K, M = u
    ρ, ω = p
    A    = A₀ * exp(-η * T^2) * (1 - κ * M)
    hill = T^n / (T^n + Tc^n)
    du[1] = α * (-λ * T + β * log(C / C★))
    du[2] = ϕ * K * (1 - M) - γ * C
    du[3] = s * A * max(K, 0.0)^θ - δ * K
    du[4] = ρ * hill * (1 - M) - ω * M          # linear damping
end

function ode_v2!(du, u, p, t)
    T, C, K, M = u
    ρ, ω = p
    A    = A₀ * exp(-η * T^2) * (1 - κ * M)
    hill = T^n / (T^n + Tc^n)
    du[1] = α * (-λ * T + β * log(C / C★))
    du[2] = ϕ * K * (1 - M) - γ * C
    du[3] = s * A * max(K, 0.0)^θ - δ * K
    du[4] = ρ * hill * (1 - M) - ω * M^2        # quadratic damping
end

function solve_model(ode!, ρ, ω; ts=tspan)
    prob = ODEProblem(ode!, float.(u0), ts, (ρ, ω))
    solve(prob, Tsit5(); saveat=1.0, reltol=1e-8, abstol=1e-10)
end

# ── Parameter grids ───────────────────────────────────────────────────────────
const ρ_grid = collect(0.05:0.05:0.50)   # 10 values
const ω_grid = collect(0.01:0.01:0.10)   # 10 values
const ω_ref  = 0.05    # reference ω for ρ sweep
const ρ_ref  = 0.25    # reference ρ for ω sweep

const var_labels = ["T (°C)", "C (ppm)", "K (relative units)", "M"]
const var_syms   = ["T", "C", "K", "M"]

# ── Plotting ──────────────────────────────────────────────────────────────────
function make_sweep_figure(ode!, sweep_vals, fixed_val, sweep_sym, fixed_sym, version_label, rho_sweep::Bool; ts=tspan)
    subplots = []
    for (vi, (vlabel, _)) in enumerate(zip(var_labels, var_syms))
        p = plot(; xlabel="Time (yr)", ylabel=vlabel, legend=:outertopright)
        for val in sweep_vals
            ρ, ω = rho_sweep ? (val, fixed_val) : (fixed_val, val)
            sol  = solve_model(ode!, ρ, ω; ts=ts)
            plot!(p, sol.t, [u[vi] for u in sol.u];
                  label="$(sweep_sym) = $(round(val; digits=3))", linewidth=1.5)
        end
        push!(subplots, p)
    end
    plot(subplots...; layout=(2, 2),
         plot_title="$(version_label) — sweeping $(sweep_sym) ($(fixed_sym) = $(fixed_val))",
         size=(1200, 800))
end

# ── Entry point ───────────────────────────────────────────────────────────────
function run_simulations()
    dir = "figures"
    isdir(dir) || mkdir(dir)

    for (ode!, label) in [(ode_v1!, "v1"), (ode_v2!, "v2")]
        fig = make_sweep_figure(ode!, ρ_grid, ω_ref, "ρ", "ω", label, true)
        savefig(fig, joinpath(dir, "$(label)_rho_sweep.pdf"))
        println("Saved $(label)_rho_sweep.pdf")

        fig = make_sweep_figure(ode!, ω_grid, ρ_ref, "ω", "ρ", label, false)
        savefig(fig, joinpath(dir, "$(label)_omega_sweep.pdf"))
        println("Saved $(label)_omega_sweep.pdf")
    end

    # Adopted model: 500-year ρ sweep at mid ω = 0.05
    fig = make_sweep_figure(ode_v1!, ρ_grid, ω_ref, "ρ", "ω", "adopted", true; ts=tspan_long)
    savefig(fig, joinpath(dir, "adopted_rho_sweep_500yr.pdf"))
    println("Saved adopted_rho_sweep_500yr.pdf")

    println("\nDone. All figures in figures/")
end

end # module climate_model_v2
