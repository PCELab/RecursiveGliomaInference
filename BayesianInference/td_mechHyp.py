# Copyright (c) 2020, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the td_hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# td_hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import ufl
import math
import numpy as np
from hippylib import *

class VectorTD():

    def __init__(self, times, tol=1e-10, mpi_comm = dl.MPI.comm_world):

        self.nsteps = len(times)
        self.data = []
        
        for i in range(self.nsteps):
            self.data.append( dl.Vector(mpi_comm) )
             
        self.times = times
        self.tol = tol
        self.mpi_comm = mpi_comm

    def __imul__(self, other):
        for d in self.data:
            d *= other
        return self
    
    def copy(self):
        res = VectorTD(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        for i in range(self.nsteps):
            res.data[i].init(self.data[i].local_range())
            res.data[i].zero()
            res.data[i].axpy(1.0, self.data[i])
        return res
    
    def initialize(self,M,dim):
        
        for d in self.data:
            M.init_vector(d,dim)
            d.zero()
            
    def axpy(self, a, other):
        for i in range(self.nsteps):
            self.data[i].axpy(a,other.data[i])
        
    def zero(self):
        for d in self.data:
            d.zero()
    
    def _find_idx(self, t):
        t_val = float(t)
        for i in range(self.nsteps):
            if abs(t_val - self.times[i]) < self.tol:
                return i
        # Fallback to midpoint logic for integration points
        i = 0
        while i < self.nsteps-1 and 2.0 * t_val > self.times[i] + self.times[i+1]:
            i += 1
        return i
            
    def store(self, u, t):
        idx = self._find_idx(t)
        self.data[idx].zero()
        self.data[idx].axpy(1., u)
        
    def retrieve(self, u, t):
        idx = self._find_idx(t)
        u.zero()
        u.axpy(1., self.data[idx])
        
    def view(self, t):
        idx = self._find_idx(t)
        return self.data[idx]      
        
    def norm(self, time_norm, space_norm):
        assert time_norm == "linf"
        s_norm = 0
        for i in range(self.nsteps):
            tmp = self.data[i].norm(space_norm)
            if tmp > s_norm:
                s_norm = tmp
        
        return s_norm
        
    def inner(self, other):
        a = 0.
        for i in range(self.nsteps):
            a += self.data[i].inner(other.data[i])
        return a

class MisfitTD(Misfit):
    def __init__(self, misfits, sim_times):
        self.misfits = misfits
        self.sim_times = sim_times

    def cost(self,x):
        c = 0.
        for itime, misfit in enumerate(self.misfits):
            c += misfit.cost([x[STATE].view(self.sim_times[itime]), x[PARAMETER], None])

        return c

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for itime, misfit in enumerate(self.misfits):
                t = self.sim_times[itime]
                misfit.grad(i, [x[STATE].view(t), x[PARAMETER], None], out.view(t))
        elif i == PARAMETER:
            out_t = out.copy()
            for itime, misfit in enumerate(self.misfits):
                out_t.zero()
                misfit.grad(i, [x[STATE].view(self.sim_times[itime]), x[PARAMETER], None], out_t)
                out.axpy(1., out_t)
        else:
            raise IndexError()

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.setLinearizationPoint([x[STATE].view(t), x[PARAMETER], None], gauss_newton_approx)

    def _apply_STATE_STATE(self, dir, out):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, STATE, dir.view(t), out.view(t))

    def _apply_STATE_PARAMETER(self, dir, out):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, PARAMETER, dir, out.view(t))

    def _apply_PARAMETER_STATE(self, dir, out):
        out_t = out.copy()
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(PARAMETER, STATE, dir.view(t), out_t)
            out.axpy(1., out_t)

    def _apply_PARAMETER_PARAMETER(self, dir, out):
        out_t = out.copy()
        for misfit in self.misfits:
            misfit.apply_ij(PARAMETER, PARAMETER, dir, out_t)
            out.axpy(1., out_t)

    def apply_ij(self,i,j,dir,out):
        out.zero()
        if i == STATE and j == STATE:
            self._apply_STATE_STATE(dir, out)
        elif i == STATE and j == PARAMETER:
            self._apply_STATE_PARAMETER(dir, out)
        elif i == PARAMETER and j == STATE:
            self._apply_PARAMETER_STATE(dir, out)
        elif i == PARAMETER and j == PARAMETER:
            self._apply_PARAMETER_PARAMETER(dir, out)
        else:
            raise 


class TimeDependentPDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear = False, is_mixed_ele = False, times=None):

        if times is not None:
            self.times = [float(t) for t in times]
        else:
            self.times = [float(t) for t in np.arange(t_init, t_final + 1)]

        self.Vh = Vh
        self.varf = varf_handler

        if isinstance(bc, dl.DirichletBC):
            self.fwd_bc = [bc]
        else:
            self.fwd_bc = bc

        if isinstance(bc0, dl.DirichletBC):
            self.adj_bc = [bc0]
        else:
            self.adj_bc = bc0

        self.mesh = self.Vh[STATE].mesh()
        self.init_cond = u0
        self.t_init = t_init
        self.t_final = t_final
        self.dt = varf_handler.dt

        # force full internal time grid exactly like linear-elastic setup
        self.times = [float(t) for t in np.arange(self.t_init, self.t_final + 0.5 * float(self.dt), float(self.dt))]
        
        self.linearize_x = None
        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.is_mixed_ele  = is_mixed_ele
        
        # Solver Params
        self.parameters = dl.NonlinearVariationalSolver.default_parameters()
        self.parameters['nonlinear_solver'] = 'snes'
        self.parameters['snes_solver']["absolute_tolerance"] = 1e-10
        self.parameters['snes_solver']["relative_tolerance"] = 1e-6
        self.parameters['snes_solver']["maximum_iterations"] = 100
        self.parameters['snes_solver']["report"] = False
        self.parameters['snes_solver']["line_search"] = "bt" # Vital for Hyper-elasticity

        # Initialize Mechanical Storage (Vector-valued)
        self.M_mech = dl.assemble(dl.inner(dl.TrialFunction(self.varf.V_vector), dl.TestFunction(self.varf.V_vector))*dl.dx)
        self.varf.u_mech_td = VectorTD(self.times)
        self.varf.u_mech_td.initialize(self.M_mech, 0)
        
        # State Mass Matrix for RD system
        self.M = dl.assemble(dl.inner(dl.TrialFunction(self.Vh[STATE]), dl.TestFunction(self.Vh[ADJOINT]))*dl.dx)

    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = VectorTD(self.times)
            u.initialize(self.M, 1)
            a = dl.Function(self.Vh[PARAMETER]).vector()
            p = VectorTD(self.times)
            p.initialize(self.M, 0)
            return [u, a, p]
        elif component == STATE:
            u = VectorTD(self.times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            return dl.Function(self.Vh[PARAMETER]).vector()
        elif component == ADJOINT:
            p = VectorTD(self.times)
            p.initialize(self.M, 0)
            return p
        else:
            raise Exception('Incorrect vector component')

    def generate_state(self):
        """ return a time dependent vector in the shape of the state """
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        """ return a time dependent vector in the shape of the adjoint """
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        """ return a time dependent vector in the shape of the state """
        u = dl.Vector()
        self.M.init_vector(u, 1)
        return u

    def generate_static_adjoint(self):
        """ return a static vector in the shape of the adjoint """
        p = dl.Vector()
        self.M.init_vector(p, 0)
        return p

    def init_parameter(self, a):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        a.init( dummy.mpi_comm(), dummy.local_range() )

    def _set_time(self, bcs, t):
        for bc in bcs:
            try:
                bc.function_arg.t = t
            except:
                pass


    def solveFwd(self, out, x):
        out.zero()
        u_old = dl.Function(self.Vh[STATE])
        u_old.vector().axpy(1.0, self.init_cond.vector())
        out.store(u_old.vector(), self.t_init)
        
        m_func = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m_func.sub(0) 

        # Initial mechanical state
        u_mech0 = self.varf.solve_mechanics(u_old, log_E_func, self.varf.log_H_fixed)
        self.varf.u_mech.vector()[:] = u_mech0.vector()
        self.varf.u_mech_td.store(u_mech0.vector(), self.t_init)

        for t in self.times[1:]:
            u_mech = self.varf.solve_mechanics(u_old, log_E_func, self.varf.log_H_fixed)
            self.varf.u_mech.vector()[:] = u_mech.vector()
            self.varf.u_mech_td.store(u_mech.vector(), t)

            u = dl.Function(self.Vh[STATE])
            res_form = self.varf(u, u_old, m_func, dl.TestFunction(self.Vh[ADJOINT]), t)
            dl.solve(res_form == 0, u, self.fwd_bc, solver_parameters=self.parameters)

            out.store(u.vector(), t)
            u_old.assign(u)

    def solveAdj(self, out, x, adj_rhs):
        out.zero()
        u, u_old = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        p = dl.Function(self.Vh[ADJOINT])
        p_next = dl.Function(self.Vh[ADJOINT])
        du, dp = dl.TestFunction(self.Vh[STATE]), dl.TrialFunction(self.Vh[ADJOINT])

        if self.solverAadj is None: 
            self.solverAadj = self._createLUSolver()
            
        p_vec = self.generate_static_adjoint() 
        rhs_t = self.generate_static_state()

        for idx in reversed(range(1, len(self.times))):
            t, t_prev = self.times[idx], self.times[idx-1]
            
            # Retrieve states
            x[STATE].retrieve(u.vector(), t)
            x[STATE].retrieve(u_old.vector(), t_prev)
            
            # Retrieve/zero RHS
            rhs_t.zero()
            adj_rhs.retrieve(rhs_t, t)
            
            # Sync Mechanics
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec
            
            # Define Adjoint Forms
            form = self.varf(u, u_old, m, p, t)
            # Jacobian w.r.t current state
            adj_form = dl.derivative(dl.derivative(form, u, du), p, dp)
            # Action of the adjoint from the next time step (back-propagation)
            b_form = dl.Constant(-1.) * dl.derivative(dl.derivative(form, u_old, du), p, p_next)
            
            Aadj, b = dl.assemble_system(adj_form, b_form, self.adj_bc)
            b.axpy(1., rhs_t)
            
            self.solverAadj.set_operator(Aadj)
            self.solverAadj.solve(p_vec, b)

            out.store(p_vec, t)
            
            # FIXED: Correct way to assign vector values in FEniCS
            p_next.vector()[:] = p_vec
    

    def exportState(self, u, fname):
        ufun = dl.Function(self.Vh[STATE], name="state")

        with dl.XDMFFile(fname) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False

            for t in self.times:
                u.retrieve(ufun.vector(), t)
                fid.write_checkpoint(ufun,"state",float(t),dl.XDMFFile.Encoding.HDF5,True)

    
    def evalGradientParameter(self, x, out):
        out.zero()
        
        # FIXED: Correct initialization of a compatible PETScVector
        out_t = dl.Vector(out.mpi_comm())
        # In FEniCS, you should use:
        out_t = dl.Vector(out) 
        
        dm = dl.TestFunction(self.Vh[PARAMETER])
        u, p, u_old = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT]), dl.Function(self.Vh[STATE])
        m_func = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m_func.sub(0)

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            x[STATE].retrieve(u.vector(), t)
            x[STATE].retrieve(u_old.vector(), t_prev)
            x[ADJOINT].retrieve(p.vector(), t)
            
            # Re-solve mechanics for consistent linearization
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec
            
            form = self.varf(u, u_old, m_func, p, t)
            
            # Reset temporary vector and assemble the partial derivative w.r.t parameter
            out_t.zero()
            dl.assemble(dl.derivative(form, m_func, dm), tensor=out_t)
            
            # Accumulate the gradient across time steps
            out.axpy(1.0, out_t)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        if self.solver_fwd_inc == None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

    def _solveIncrementalFwd(self, out, rhs):
        out.zero()
        u, u_old = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)

        dp, du = dl.TestFunction(self.Vh[ADJOINT]), dl.TrialFunction(self.Vh[STATE])
        uhat_vec = self.generate_static_state()
        uhat_old = dl.Function(self.Vh[STATE])
        uhat_old.vector().zero()

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            
            # Re-sync mechanics for linearization
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec
            
            form = self.varf(u, u_old, m, dp, t)
            # Jacobian w.r.t current state and action of Jacobian w.r.t previous state
            Ainc_form = dl.derivative(form, u, du)
            rhs_prev_form = dl.Constant(-1.) * dl.derivative(form, u_old, uhat_old)
            
            Ainc, binc = dl.assemble_system(Ainc_form, rhs_prev_form, self.fwd_bc)
            binc.axpy(1., rhs.view(t))

            self.solver_fwd_inc.set_operator(Ainc)
            self.solver_fwd_inc.solve(uhat_vec, binc)

            out.store(uhat_vec, t)
            
            # FIXED: Use slicing for vector assignment
            uhat_old.vector()[:] = uhat_vec

    def _solveIncrementalAdj(self, out, rhs):
        out.zero()
        u, u_old = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        p = dl.Function(self.Vh[ADJOINT])
        
        phat_vec = self.generate_static_adjoint()
        phat_next = dl.Function(self.Vh[ADJOINT])
        phat_next.vector().zero()

        du, dp = dl.TestFunction(self.Vh[STATE]), dl.TrialFunction(self.Vh[ADJOINT])

        for idx in reversed(range(1, len(self.times))):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec

            form = self.varf(u, u_old, m, p, t)
            # Adjoint Jacobian and action of back-propagated adjoint
            A_adj_form = dl.derivative(dl.derivative(form, u, du), p, dp)
            rhs_next_form = dl.Constant(-1.) * dl.derivative(dl.derivative(form, u_old, du), p, phat_next)
            
            A_adj_inc, b_adj_inc = dl.assemble_system(A_adj_form, rhs_next_form, self.adj_bc)
            b_adj_inc.axpy(1., rhs.view(t))
            
            self.solver_adj_inc.set_operator(A_adj_inc)
            self.solver_adj_inc.solve(phat_vec, b_adj_inc)

            out.store(phat_vec, t)
            
            # FIXED: Use slicing for vector assignment
            phat_next.vector()[:] = phat_vec


    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.

            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            return self._solveIncrementalAdj(out, rhs)
        else:
            return self._solveIncrementalFwd(out, rhs)


    def applyC(self, dm, out):
        out.zero()
        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        dp_test = dl.TestFunction(self.Vh[ADJOINT])
        
        # Pre-allocate assembly vector for adjoint-sized output
        out_t = self.generate_static_adjoint()

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            # Sync mechanics for correct geometry
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec
            
            form = self.varf(u, u_old, m, p, t)
            # Jacobian of PDE w.r.t parameter: d/dm ( d/dp * form )
            cvarf = dl.derivative(dl.derivative(form, p, dp_test), m, dm_fun)
            
            out_t.zero()
            dl.assemble(cvarf, tensor=out_t)
            # Apply homogeneous adjoint BCs
            [bc.apply(out_t) for bc in self.adj_bc]
            out.store(out_t, t)

    def applyCt(self, dp, out):
        out.zero()
        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        
        dp_fun = dl.Function(self.Vh[ADJOINT])
        dm_test = dl.TestFunction(self.Vh[PARAMETER])
        
        # FIXED: Use the 'out' vector as a prototype to ensure correct MPI partitioning
        out_p = dl.Vector(out)

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            # Retrieve the adjoint direction (dp) for the current time step
            dp.retrieve(dp_fun.vector(), t)
            
            # Sync mechanics for correct geometry
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec
            
            form = self.varf(u, u_old, m, p, t)
            
            # Transpose Jacobian: d/dm ( d/dp * form * dp_fun )
            cvarf_adj = dl.derivative(dl.derivative(form, p, dp_fun), m, dm_test)
            
            out_p.zero()
            dl.assemble(cvarf_adj, tensor=out_p)
            
            # Accumulate into the parameter-space output
            out.axpy(1.0, out_p)


    def applyWuu(self, du, out):
        out.zero()
        if self.gauss_newton_approx:
            return

        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)

        du_curr, du_prev = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE])
        v_test = dl.TestFunction(self.Vh[STATE])
        
        # Pre-allocate assembly vector
        out_t = self.generate_static_adjoint()

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            du.retrieve(du_curr.vector(), t)
            du.retrieve(du_prev.vector(), t_prev)

            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec

            form = self.varf(u, u_old, m, p, t)
            # Second derivative of Lagrangian w.r.t state
            Luu_form = dl.derivative(dl.derivative(form, u, du_curr), u, v_test) + \
                       dl.derivative(dl.derivative(form, u_old, du_prev), u, v_test)

            out_t.zero()
            dl.assemble(Luu_form, tensor=out_t)
            [bc.apply(out_t) for bc in self.adj_bc]
            out.store(out_t, t)

    def applyWum(self, dm, out):
        out.zero()
        if self.gauss_newton_approx:
            return

        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        du_test = dl.TestFunction(self.Vh[STATE])
        
        # Pre-allocate assembly vector for the state/adjoint space
        out_t = self.generate_static_adjoint()

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            # Re-sync Hyper-elasticity for the current time step
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec

            form = self.varf(u, u_old, m, p, t)
            
            # Mixed derivative: d/du ( d/dm * form * dm_fun )
            # Note: We take derivative w.r.t current state 'u' and previous state 'u_old'
            # to ensure the full temporal coupling is captured.
            L_um_form = dl.derivative(dl.derivative(form, m, dm_fun), u, du_test) + \
                        dl.derivative(dl.derivative(form, m, dm_fun), u_old, du_test)

            out_t.zero()
            dl.assemble(L_um_form, tensor=out_t)
            
            # Apply homogeneous Dirichlet BCs (Adjoint/State space)
            [bc.apply(out_t) for bc in self.adj_bc]
            
            out.store(out_t, t)
    
    def applyWmu(self, du, out):
        out.zero()
        if self.gauss_newton_approx:
            return

        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        
        du_curr, du_prev = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE])
        dm_test = dl.TestFunction(self.Vh[PARAMETER])
        
        # FIXED: Initialize out_t as a twin of out to ensure 
        # identical MPI partitioning and PETSc layout.
        out_t = dl.Vector(out) 

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)
            
            # Retrieve the trial directions for the current and previous states
            du.retrieve(du_curr.vector(), t)
            du.retrieve(du_prev.vector(), t_prev)

            # Synchronize the mechanical solver state
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec

            form = self.varf(u, u_old, m, p, t)
            
            # Mixed derivative: d/dm [ (d/du * form * du_curr) + (d/du_old * form * du_prev) ]
            Lmu_form = dl.derivative(dl.derivative(form, u, du_curr), m, dm_test) + \
                       dl.derivative(dl.derivative(form, u_old, du_prev), m, dm_test)

            out_t.zero()
            dl.assemble(Lmu_form, tensor=out_t)
            out.axpy(1.0, out_t)

    def applyWmm(self, dm, out):
        out.zero()
        if self.gauss_newton_approx:
            return

        u, u_old, p = dl.Function(self.Vh[STATE]), dl.Function(self.Vh[STATE]), dl.Function(self.Vh[ADJOINT])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        log_E_func = m.sub(0)
        
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        dm_test = dl.TestFunction(self.Vh[PARAMETER])
        
        # FIXED: Correct way to initialize a compatible vector in FEniCS
        out_t = dl.Vector(out) 

        for idx in range(1, len(self.times)):
            t, t_prev = self.times[idx], self.times[idx-1]
            self.linearize_x[STATE].retrieve(u.vector(), t)
            self.linearize_x[STATE].retrieve(u_old.vector(), t_prev)
            self.linearize_x[ADJOINT].retrieve(p.vector(), t)

            # Re-sync mechanics for the correct geometry at this time step
            u_mech_vec = self.varf.u_mech_td.view(t)
            self.varf.u_mech.vector()[:] = u_mech_vec

            form = self.varf(u, u_old, m, p, t)
            
            # Second derivative of Lagrangian w.r.t parameter
            Lmm_form = dl.derivative(dl.derivative(form, m, dm_fun), m, dm_test)
            
            out_t.zero()
            dl.assemble(Lmm_form, tensor=out_t)
            out.axpy(1.0, out_t)

    def apply_ij(self,i,j, dir, out):
        """
            Given u, a, p; compute
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.applyWuu
        KKT[PARAMETER, STATE] = self.applyWmu
        KKT[STATE, PARAMETER] = self.applyWum
        KKT[PARAMETER, PARAMETER] = self.applyWmm
        KKT[ADJOINT, STATE] = None
        KKT[STATE, ADJOINT] = None

        KKT[ADJOINT, PARAMETER] = self.applyC
        KKT[PARAMETER, ADJOINT] = self.applyCt
        KKT[i,j](dir, out)


    def _createLUSolver(self):
        return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )
           