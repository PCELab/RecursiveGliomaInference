# mech_hyper.py

import dolfin as dl
import ufl
import math
from fenics import *

STATE, PARAMETER, ADJOINT = 0, 1, 2

class MechCoupledTumorVarf:
    def __init__(self, Vh, dt, dx, log_H_fixed, c=0.65, nu=0.45):
        self.dt = Constant(1.0 / dt)
        self.dx = dx
        self.V_scalar = Vh[STATE]
        self.V_vector = VectorFunctionSpace(self.V_scalar.mesh(), "Lagrange", 2)

        # Biomechanical parameters
        self.nu = Constant(nu)
        self.c = Constant(c)
        self.log_H_fixed = log_H_fixed

        self.zero = Constant((0.0, 0.0))
        self.bc_mech = DirichletBC(self.V_vector, self.zero, "on_boundary")
        
        # MODIFICATION: Initialize u_mech as a zero function immediately
        # This ensures that at t=0, displacement and stress-dependent terms are null.
        self.u_mech = Function(self.V_vector, name="u_mech")
        self.u_mech.vector()[:] = 0.0 

    # ------------------------------------------------------------------
    # Mechanics storage (for time-dependent problems)
    # ------------------------------------------------------------------
    def set_mechanical_displacement(self, u_sol, time=None):
        if time is not None:
            if not hasattr(self, "u_mech_td"):
                raise RuntimeError("u_mech_td not initialized. "
                                   "Call initialize_mechanical_storage(times, Vh_vector) first.")
            u_vec = dl.Vector()
            u_vec.init(u_sol.vector().local_range())
            u_vec.axpy(1.0, u_sol.vector())
            self.u_mech_td.store(u_vec, time)
        else:
            self.u_mech = u_sol  # static mechanics

    def get_mechanical_displacement(self, time=None):
        if hasattr(self, "u_mech_td") and time is not None:
            u_vec = self.u_mech_td.view(time)
            u_func = dl.Function(self.V_vector, name="u_mech")
            u_func.vector().set_local(u_vec.get_local())
            return u_func
        else:
            return self.u_mech

    # ------------------------------------------------------------------
    # Hyperelastic constitutive law (PHYSICS UNTOUCHED)
    # ------------------------------------------------------------------
    def strain_energy(self, u, log_E):
        E_val = ufl.exp(log_E)
        mu = E_val / (2.0 * (1.0 + self.nu))
        K_bulk = E_val / (3.0 * (1.0 - 2.0 * self.nu))
        F2 = ufl.Identity(2) + ufl.grad(u)
        F = ufl.as_tensor([[F2[0, 0], F2[0, 1], 0.0],
                           [F2[1, 0], F2[1, 1], 0.0],
                           [0.0,       0.0,      1.0]])
        J = ufl.det(F)
        B = F * ufl.transpose(F)
        I1 = ufl.tr(B)
        I1_bar = J**(-2.0 / 3.0) * I1
        W = (mu / 2.0) * (I1_bar - 3.0) + (K_bulk / 2.0) * (J - 1.0)**2
        return W

    def first_piola_from_energy(self, u, log_E):
        P3 = self.first_piola_3x3(u, log_E)
        P2 = ufl.as_tensor([[P3[0, 0], P3[0, 1]],
                           [P3[1, 0], P3[1, 1]]])
        return P2

    def first_piola_3x3(self, u, log_E):
        E_val = ufl.exp(log_E)
        mu = E_val / (2.0 * (1.0 + self.nu))
        K_bulk = E_val / (3.0 * (1.0 - 2.0 * self.nu))
        F2 = ufl.Identity(2) + ufl.grad(u)
        F = ufl.as_tensor([[F2[0, 0], F2[0, 1], 0.0],
                           [F2[1, 0], F2[1, 1], 0.0],
                           [0.0,       0.0,      1.0]])
        J = ufl.det(F)
        B = F * ufl.transpose(F)
        I1 = ufl.tr(B)
        FinvT = ufl.transpose(ufl.inv(F))
        P_iso = mu * J**(-2.0/3.0) * (F - (1.0 / 3.0) * I1 * FinvT)
        P_vol = K_bulk * (J - 1.0) * J * FinvT
        return P_iso + P_vol

    def cauchy_from_P(self, P3, F3):
        J = ufl.det(F3)
        return (1.0 / J) * (P3 * ufl.transpose(F3))

    def von_mises_from_cauchy(self, T3):
        trT = ufl.tr(T3)
        s = T3 - (1.0 / 3.0) * trT * ufl.Identity(3)
        return ufl.sqrt((3.0 / 2.0) * ufl.inner(s, s))

    def solve_mechanics(self, phi_func, log_E, log_H):
        # 1. Setup Functions
        u = dl.Function(self.V_vector, name="Displacement")
        u.vector()[:] = 0.0  # Start from zero displacement for stability
        
        w = dl.TestFunction(self.V_vector)
        du = dl.TrialFunction(self.V_vector) # <--- The missing TrialFunction for the Jacobian

        # 2. Kinematics & Physical Constraints
        # We use ufl.variable so we can differentiate the energy functional
        F2 = ufl.variable(ufl.Identity(2) + ufl.grad(u))
        
        # 3D reconstruction for volumetric consistency (Plane Strain)
        F3 = ufl.as_tensor([[F2[0, 0], F2[0, 1], 0.0],
                            [F2[1, 0], F2[1, 1], 0.0],
                            [0.0,       0.0,       1.0]])
        
        J3 = ufl.det(F3)
        
        # Stabilization: Prevent J from becoming non-positive during Newton iterations
        # This is often why hyper-elastic models produce NaNs
        J3_safe = ufl.conditional(ufl.le(J3, 1e-4), 1e-4, J3)
        
        FinvT3 = ufl.transpose(ufl.inv(F3))
        FinvT_2x2 = ufl.as_tensor([[FinvT3[0, 0], FinvT3[0, 1]],
                                   [FinvT3[1, 0], FinvT3[1, 1]]])

        # 3. Weak Form (Residual)
        # External force from tumor growth pressure
        grad_phi_ref = ufl.grad(phi_func)
        b0_vec = self.c * J3_safe * (FinvT_2x2 * grad_phi_ref)   

        # Internal forces from Hyper-elastic energy
        P2 = self.first_piola_from_energy(u, log_E)   
        
        # Total Residual
        F_mech = ufl.inner(P2, ufl.grad(w)) * self.dx - ufl.dot(b0_vec, w) * self.dx
        
        # 4. Jacobian Definition
        # This is where 'du' is used. The solver needs this for the linear sub-steps.
        J_mech = ufl.derivative(F_mech, u, du)

        # 5. Nonlinear Solve
        # Using 'snes' as requested in your setup
        dl.solve(F_mech == 0, u, self.bc_mech, J=J_mech,
              solver_parameters={
                  "nonlinear_solver": "snes",
                  "snes_solver": {
                      "maximum_iterations": 50,
                      "relative_tolerance": 1e-7,
                      "absolute_tolerance": 1e-9,
                      "report": False,
                      "error_on_nonconvergence": False,
                      "line_search": "bt" # Backtracking line search helps with NaNs
                  }
              })
        
        return u

    # ------------------------------------------------------------------
    # Reaction–diffusion with mechanics
    # ------------------------------------------------------------------

    def __call__(self, u, u_old, m, p, t):
        # 1. Parameter Splitting
        log_E, log_D, log_G = split(m)
        G = ufl.exp(log_G)

        # 2. Mechanical Displacement Handling
        # We use the persistent self.u_mech which is updated by the 
        # TimeDependentPDEVariationalProblem solver before this call.
        # This ensures the symbolic graph remains intact for Adjoint/Hessian.
        u_mech = self.u_mech

        # --- PHYSICS CALCULATIONS (Kinematics) ---
        # Identity and Deformation Gradient in 2D (mapped to 3D for J and Piola)
        F2_mech = ufl.Identity(2) + ufl.grad(u_mech)
        F3 = ufl.as_tensor([[F2_mech[0, 0], F2_mech[0, 1], 0.0],
                           [F2_mech[1, 0], F2_mech[1, 1], 0.0],
                           [0.0,            0.0,           1.0]])
        J3 = ufl.det(F3)

        # Stress and Constitutive relation
        P3_mech = self.first_piola_3x3(u_mech, log_E)   
        T3 = self.cauchy_from_P(P3_mech, F3)
        vm = self.von_mises_from_cauchy(T3)

        # 3. Mechanical Inhibition of Diffusion
        H_D = ufl.exp(self.log_H_fixed)
        D_val = ufl.exp(log_D) * ufl.exp(-H_D * vm)

        # Inverse deformation for gradient mapping (Pull-back to reference)
        FinvT3 = ufl.transpose(ufl.inv(F3))
        FinvT_2x2 = ufl.as_tensor([[FinvT3[0, 0], FinvT3[0, 1]],
                                   [FinvT3[1, 0], FinvT3[1, 1]]])

        # --- WEAK FORM ASSEMBLY ---
        # Time derivative term (scaled by Jacobian J3 to account for reference area)
        time_term = J3 * (u - u_old) * self.dt * p * self.dx
        
        # Diffusion term: J * D * (invF * grad_u) . (invF * grad_p)
        diff_grad_u = FinvT_2x2 * ufl.grad(u)
        diff_grad_p = FinvT_2x2 * ufl.grad(p)
        diff_term = J3 * D_val * ufl.inner(diff_grad_u, diff_grad_p) * self.dx
        
        # Reaction term (Logistic growth)
        react_term = - J3 * G * (1.0 - u_old) * u_old * p * self.dx

        return time_term + diff_term + react_term