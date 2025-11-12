"""
Single-Stage Axial Fan Design Tool
Author: CJC
Date: November 5, 2025

This module designs a single-stage axial fan based on specified parameters
and calculates velocity triangles, blade angles, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


class AxialFanDesign:
    """
    Single-stage axial fan design calculator
    """
    
    def __init__(self, rhub, rcas, Cx, Nblade, Beta_1m, vort_exp, rpm, DeHaller):
        """
        Initialize fan design parameters
        
        Parameters:
        -----------
        rhub : float
            Hub radius (m)
        rcas : float
            Casing/tip radius (m)
        Cx : float
            Axial chord length (m)
        Nblade : int
            Number of blades
        Beta_1m : float
            RELATIVE inlet flow angle at mean radius (degrees)
        vort_exp : float
            Vortex distribution exponent (n) applied to ABSOLUTE tangential velocity
            -1: free vortex, 0: constant, 1: forced vortex
        rpm : float
            Rotational speed (revolutions per minute)
        DeHaller : float
            De Haller number at midspan (W2/W1), typically 0.72-0.75
        """
        self.rhub = rhub
        self.rcas = rcas
        self.Cx = Cx
        self.Nblade = Nblade
        self.Beta_1m = Beta_1m  # degrees (relative angle)
        self.vort_exp = vort_exp
        self.rpm = rpm
        self.DeHaller = DeHaller
        
        # Calculate derived geometric parameters
        self.rmean = (rhub + rcas) / 2
        self.hub_tip_ratio = rhub / rcas
        self.omega = 2 * np.pi * rpm / 60  # rad/s
        
        # Initialize results storage
        self.results = {}
        
    def calculate_design(self, n_points=50):
        """
        Perform complete fan design calculation
        
        Parameters:
        -----------
        n_points : int
            Number of radial stations for analysis
        """
        # Radial distribution from hub to tip
        r = np.linspace(self.rhub, self.rcas, n_points)
        
        # Blade speed at each radius
        U = self.omega * r
        U_mean = self.omega * self.rmean
        
        # INLET CONDITIONS
        # Absolute inlet: purely axial (Alpha_1 = 0° everywhere)
        Vtheta_1 = np.zeros_like(r)  # No swirl at inlet
        Alpha_1 = np.zeros_like(r)   # Pure axial inlet
        
        # Assume constant axial velocity throughout
        # Calculate from mean radius relative inlet angle
        # Convention: Beta measured from axial direction (Beta=0° is axial)
        # tan(Beta) = Wtheta / Vx
        Vx = U_mean / np.tan(np.radians(self.Beta_1m))
        Vx_array = np.ones_like(r) * Vx
        
        # Inlet relative velocity components
        # Wtheta = Vtheta - U = 0 - U = -U (since Vtheta_1 = 0)
        Wtheta_1 = Vtheta_1 - U  
        W_1 = np.sqrt(Vx_array**2 + Wtheta_1**2)
        # Beta measured from axial: tan(Beta) = Wtheta / Vx
        # Positive when Wtheta > 0 (flow in direction of rotation)
        # Negative when Wtheta < 0 (flow opposite to rotation)
        Beta_1 = np.degrees(np.arctan(Wtheta_1 / Vx_array))
        
        # OUTLET CONDITIONS AT MEAN
        # Use De Haller number to set W2 at mean
        # De Haller = W2/W1 < 1 (diffusion in relative frame)
        idx_mean = n_points // 2
        W_1_mean = W_1[idx_mean]
        W_2_mean_requested = self.DeHaller * W_1_mean
        
        # Maximum possible De Haller occurs when flow turns to purely axial (β2 = 0°)
        # At that point, W2 = Vx (Wtheta_2 = 0)
        DeHaller_max = Vx / W_1_mean
        
        # Check if requested De Haller is achievable
        if self.DeHaller < DeHaller_max:
            print(f"\n⚠ WARNING: Requested De Haller ({self.DeHaller:.3f}) < minimum achievable ({DeHaller_max:.3f})")
            print(f"           Cannot turn flow beyond axial in relative frame!")
            print(f"           Setting to maximum turning: β2 = 0° (purely axial relative outlet)")
            W_2_mean = Vx
            Wtheta_2_mean = 0.0
            DeHaller_actual = DeHaller_max
            print(f"           Actual De Haller at mean: {DeHaller_actual:.3f}\n")
        else:
            W_2_mean = W_2_mean_requested
            # Wtheta_2 is still negative but smaller magnitude than Wtheta_1
            Wtheta_2_mean = -np.sqrt(W_2_mean**2 - Vx**2)  
            DeHaller_actual = self.DeHaller
        
        # Beta measured from axial: tan(Beta) = Wtheta / Vx (signed)
        Beta_2_mean = np.degrees(np.arctan(Wtheta_2_mean / Vx))
        
        # Calculate absolute tangential velocity at mean outlet
        # Wtheta = Vtheta - U, so Vtheta = Wtheta + U
        Vtheta_2_mean = Wtheta_2_mean + U_mean
        
        # APPLY VORTEX LAW TO OUTLET ABSOLUTE TANGENTIAL VELOCITY
        # r^n * Vtheta = constant = r_mean^n * Vtheta_2_mean
        C_vortex = (self.rmean ** self.vort_exp) * Vtheta_2_mean
        Vtheta_2 = C_vortex / (r ** self.vort_exp)
        
        # Outlet absolute flow angles (measured from axial)
        # tan(Alpha) = Vtheta / Vx
        Alpha_2 = np.degrees(np.arctan2(Vtheta_2, Vx_array))
        
        # Outlet relative velocity components
        Wtheta_2 = Vtheta_2 - U
        W_2 = np.sqrt(Vx_array**2 + Wtheta_2**2)
        # Beta measured from axial: tan(Beta) = Wtheta / Vx (signed)
        # Positive when Wtheta > 0 (flow in direction of rotation)
        # Negative when Wtheta < 0 (flow opposite to rotation)
        # Angle can cross through zero when relative flow crosses axial
        Beta_2 = np.degrees(np.arctan(Wtheta_2 / Vx_array))
        
        # Flow turning (in relative frame)
        # Delta_Beta = Beta_1 - Beta_2
        # Positive: turning towards axial (typical compressor)
        # Can be negative if over-turned past axial
        Delta_Beta = Beta_1 - Beta_2
        
        # Work and loading (Euler turbine equation)
        Delta_h = U * (Vtheta_2 - Vtheta_1)  # Since Vtheta_1 = 0, Delta_h = U * Vtheta_2
        
        # Absolute velocities
        V_1 = Vx_array  # Since no inlet swirl
        V_2 = np.sqrt(Vx_array**2 + Vtheta_2**2)
        
        # De Haller number distribution
        DeHaller_dist = W_2 / W_1
        
        # Solidity and blade spacing
        spacing = 2 * np.pi * r / self.Nblade
        
        # Stagger angle (mean of inlet and outlet relative angles)
        stagger = 0.5 * (Beta_1 + Beta_2)  # degrees
        
        # True chord (accounting for stagger)
        # C_true = C_axial / cos(stagger)
        C_true = self.Cx / np.cos(np.radians(stagger))
        
        # Solidity (true chord / spacing)
        solidity = C_true / spacing
        
        # Lieblein Diffusion Factor
        # D = 1 - (W2/W1) + (ΔVθ / (2*σ*W1))
        Delta_Vtheta = Vtheta_2 - Vtheta_1  # Since Vtheta_1 = 0, this equals Vtheta_2
        diffusion_factor = 1 - DeHaller_dist + (np.abs(Delta_Vtheta) / (2 * solidity * W_1))
        
        # PERFORMANCE CALCULATIONS (at atmospheric conditions)
        rho = 1.225  # kg/m³ (air density at sea level, 15°C)
        
        # Isentropic (stagnation) pressure rise: Δp0 = ρ * Δh
        Delta_p0 = rho * Delta_h  # Pa
        
        # Static pressure at exit (using Bernoulli from mean conditions)
        # p0_exit - p_exit = 0.5 * ρ * V_exit²
        # p0_exit = p0_inlet + Δp0
        # Assume p0_inlet ≈ p_inlet (inlet is axial, so V_inlet ≈ Vx, dynamic head small)
        # Therefore: p_exit ≈ p_inlet + Δp0 - 0.5 * ρ * V_exit²
        V_exit_squared = Vx_array**2 + Vtheta_2**2
        Delta_p_static = Delta_p0 - 0.5 * rho * V_exit_squared
        
        # Mass flow rate
        # m_dot = ρ * Vx * A_annulus
        A_annulus = np.pi * (self.rcas**2 - self.rhub**2)  # m²
        m_dot = rho * Vx * A_annulus  # kg/s
        
        # Store results
        self.results = {
            'r': r,
            'U': U,
            'Vx': Vx_array,
            'V_1': V_1,
            'V_2': V_2,
            'Vtheta_1': Vtheta_1,
            'Vtheta_2': Vtheta_2,
            'Alpha_1': Alpha_1,
            'Alpha_2': Alpha_2,
            'Beta_1': Beta_1,
            'Beta_2': Beta_2,
            'W_1': W_1,
            'W_2': W_2,
            'Delta_Beta': Delta_Beta,
            'Delta_h': Delta_h,
            'DeHaller': DeHaller_dist,
            'stagger': stagger,
            'C_true': C_true,
            'spacing': spacing,
            'solidity': solidity,
            'diffusion_factor': diffusion_factor,
            'Delta_p0': Delta_p0,
            'Delta_p_static': Delta_p_static,
            'rho': rho,
            'm_dot': m_dot,
            'A_annulus': A_annulus
        }
        
        return self.results
    
    def plot_velocity_triangles(self, radii_fraction=[0.0, 0.5, 1.0]):
        """
        Plot velocity triangles at specified radial locations
        
        Parameters:
        -----------
        radii_fraction : list
            Fractional positions along span (0=hub, 1=tip)
        """
        if not self.results:
            print("Run calculate_design() first!")
            return
        
        r = self.results['r']
        n_points = len(r)
        
        fig, axes = plt.subplots(1, len(radii_fraction), figsize=(15, 4))
        if len(radii_fraction) == 1:
            axes = [axes]
        
        for idx, frac in enumerate(radii_fraction):
            i = int(frac * (n_points - 1))
            ax = axes[idx]
            
            # Get velocities at this station
            U = self.results['U'][i]
            Vx = self.results['Vx'][i]
            Vtheta_1 = self.results['Vtheta_1'][i]
            Vtheta_2 = self.results['Vtheta_2'][i]
            
            # Inlet triangle
            ax.arrow(0, 0, Vtheta_1, Vx, head_width=2, head_length=2, 
                    fc='blue', ec='blue', label='V1 (abs)')
            ax.arrow(0, 0, U, 0, head_width=2, head_length=2, 
                    fc='red', ec='red', label='U')
            ax.arrow(U, 0, Vtheta_1-U, Vx, head_width=2, head_length=2, 
                    fc='green', ec='green', label='W1 (rel)')
            
            # Outlet triangle (offset for clarity)
            offset = U * 0.2
            ax.arrow(offset, -20, Vtheta_2, Vx, head_width=2, head_length=2, 
                    fc='blue', ec='blue', linestyle='--', label='V2 (abs)')
            ax.arrow(offset, -20, U, 0, head_width=2, head_length=2, 
                    fc='red', ec='red', linestyle='--')
            ax.arrow(U+offset, -20, Vtheta_2-U, Vx, head_width=2, head_length=2, 
                    fc='green', ec='green', linestyle='--', label='W2 (rel)')
            
            location = ['Hub', 'Mean', 'Tip'][idx] if frac in [0.0, 0.5, 1.0] else f'{frac:.1%} span'
            ax.set_title(f'{location} (r={r[i]:.3f} m)')
            ax.set_xlabel('Tangential (m/s)')
            ax.set_ylabel('Axial (m/s)')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            if idx == 0:
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('figs/velocity_triangles.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_radial_distributions(self):
        """
        Plot radial distributions of key parameters
        """
        if not self.results:
            print("Run calculate_design() first!")
            return
        
        r = self.results['r']
        r_norm = (r - self.rhub) / (self.rcas - self.rhub)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Blade angles
        axes[0, 0].plot(r_norm, self.results['Beta_1'], 'b-', label='β1 (inlet)')
        axes[0, 0].plot(r_norm, self.results['Beta_2'], 'r-', label='β2 (outlet)')
        axes[0, 0].set_xlabel('Normalized Radius')
        axes[0, 0].set_ylabel('Blade Angle (deg)')
        axes[0, 0].set_title('Relative Flow Angles')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Flow angles
        axes[0, 1].plot(r_norm, self.results['Alpha_1'], 'b-', label='α1 (inlet)')
        axes[0, 1].plot(r_norm, self.results['Alpha_2'], 'r-', label='α2 (outlet)')
        axes[0, 1].set_xlabel('Normalized Radius')
        axes[0, 1].set_ylabel('Flow Angle (deg)')
        axes[0, 1].set_title('Absolute Flow Angles')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Flow turning
        axes[0, 2].plot(r_norm, self.results['Delta_Beta'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Normalized Radius')
        axes[0, 2].set_ylabel('Δβ (deg)')
        axes[0, 2].set_title('Flow Turning')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Velocities
        axes[1, 0].plot(r_norm, self.results['U'], 'r-', label='U (blade speed)')
        axes[1, 0].plot(r_norm, self.results['W_1'], 'b-', label='W1 (rel inlet)')
        axes[1, 0].plot(r_norm, self.results['W_2'], 'g-', label='W2 (rel outlet)')
        axes[1, 0].set_xlabel('Normalized Radius')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].set_title('Velocities')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # De Haller number
        axes[1, 1].plot(r_norm, self.results['DeHaller'], 'orange', linewidth=2)
        axes[1, 1].axhline(y=0.72, color='r', linestyle='--', alpha=0.5, label='Min limit (0.72)')
        axes[1, 1].set_xlabel('Normalized Radius')
        axes[1, 1].set_ylabel('De Haller (W2/W1)')
        axes[1, 1].set_title('De Haller Number')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Lieblein Diffusion Factor
        axes[1, 2].plot(r_norm, self.results['diffusion_factor'], 'brown', linewidth=2)
        axes[1, 2].axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Max limit (0.6)')
        axes[1, 2].set_xlabel('Normalized Radius')
        axes[1, 2].set_ylabel('Diffusion Factor')
        axes[1, 2].set_title('Lieblein Diffusion Factor')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figs/radial_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional plot for pressure distribution
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        
        # Pressure rises
        ax2.plot(r_norm, self.results['Delta_p0'], 'b-', linewidth=2, label='Stagnation Δp₀')
        ax2.plot(r_norm, self.results['Delta_p_static'], 'r-', linewidth=2, label='Static Δp')
        ax2.set_xlabel('Normalized Radius')
        ax2.set_ylabel('Pressure Rise (Pa)')
        ax2.set_title('Pressure Rise Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figs/pressure_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """
        Print design summary
        """
        print("\n" + "="*60)
        print("AXIAL FAN DESIGN SUMMARY")
        print("="*60)
        print(f"\nGEOMETRIC PARAMETERS:")
        print(f"  Hub radius:              {self.rhub:.4f} m")
        print(f"  Tip radius:              {self.rcas:.4f} m")
        print(f"  Mean radius:             {self.rmean:.4f} m")
        print(f"  Hub-to-tip ratio:        {self.hub_tip_ratio:.3f}")
        print(f"  Axial chord:             {self.Cx:.4f} m")
        print(f"  Number of blades:        {self.Nblade}")
        print(f"\nOPERATING CONDITIONS:")
        print(f"  Rotational speed:        {self.rpm:.1f} rpm")
        print(f"  Angular velocity:        {self.omega:.2f} rad/s")
        print(f"  Vortex exponent:         {self.vort_exp:.2f}")
        print(f"  Rel. inlet angle (mean): {self.Beta_1m:.1f}°")
        print(f"  De Haller (design):      {self.DeHaller:.3f}")
        
        if self.results:
            # Find hub, mean, tip indices
            n = len(self.results['r'])
            idx_hub = 0
            idx_mean = n // 2
            idx_tip = n - 1
            
            print(f"\nBLADE SPEED:")
            print(f"  At hub:                  {self.results['U'][idx_hub]:.2f} m/s")
            print(f"  At mean:                 {self.results['U'][idx_mean]:.2f} m/s")
            print(f"  At tip:                  {self.results['U'][idx_tip]:.2f} m/s")
            
            print(f"\nFLOW ANGLES AT MEAN:")
            print(f"  α1 (inlet abs):          {self.results['Alpha_1'][idx_mean]:.2f}°")
            print(f"  α2 (outlet abs):         {self.results['Alpha_2'][idx_mean]:.2f}°")
            print(f"  β1 (inlet rel):          {self.results['Beta_1'][idx_mean]:.2f}°")
            print(f"  β2 (outlet rel):         {self.results['Beta_2'][idx_mean]:.2f}°")
            print(f"  Flow turning (Δβ):       {self.results['Delta_Beta'][idx_mean]:.2f}°")
            
            print(f"\nPERFORMANCE AT MEAN:")
            print(f"  Specific work:           {self.results['Delta_h'][idx_mean]/1000:.3f} kJ/kg")
            print(f"  De Haller (actual):      {self.results['DeHaller'][idx_mean]:.3f}")
            print(f"  De Haller (minimum):     {self.results['Vx'][idx_mean]/self.results['W_1'][idx_mean]:.3f}")
            print(f"  Diffusion factor:        {self.results['diffusion_factor'][idx_mean]:.3f}")
            print(f"  Stagnation pressure rise:{self.results['Delta_p0'][idx_mean]:.2f} Pa")
            print(f"  Static pressure rise:    {self.results['Delta_p_static'][idx_mean]:.2f} Pa")
            print(f"  Stagger angle:           {self.results['stagger'][idx_mean]:.2f}°")
            print(f"  True chord:              {self.results['C_true'][idx_mean]*1000:.2f} mm")
            print(f"  Blade spacing:           {self.results['spacing'][idx_mean]*1000:.2f} mm")
            print(f"  Solidity (true chord):   {self.results['solidity'][idx_mean]:.3f}")
            
            print(f"\nOVERALL PERFORMANCE:")
            print(f"  Air density:             {self.results['rho']:.3f} kg/m³")
            print(f"  Annulus area:            {self.results['A_annulus']*1e6:.2f} mm²")
            print(f"  Axial velocity:          {self.results['Vx'][idx_mean]:.2f} m/s")
            print(f"  Mass flow rate:          {self.results['m_dot']:.6f} kg/s")
            print(f"  Volume flow rate:        {self.results['m_dot']/self.results['rho']*1000:.3f} L/s")
            print(f"  Volume flow rate:        {self.results['m_dot']/self.results['rho']*3600:.2f} m³/h")
        
        print("="*60 + "\n")
    
    def generate_characteristic(self, flow_coefficients=None, n_points=50):
        """
        Generate fan characteristic curves by varying axial velocity
        while keeping outlet angles fixed (design point)
        
        Parameters:
        -----------
        flow_coefficients : array-like or None
            Array of flow coefficients (Vx/U_mean) to evaluate
            If None, generates from 0.3 to 1.2 times design point
        n_points : int
            Number of radial stations for each operating point
        
        Returns:
        --------
        dict : Characteristic data including arrays of flow rate, pressure rise, etc.
        """
        # First, calculate design point to get reference values
        self.calculate_design(n_points=n_points)
        
        # Store design point outlet angles (these remain fixed)
        Beta_2_design = self.results['Beta_2'].copy()
        Vtheta_2_design = self.results['Vtheta_2'].copy()
        Vx_design = self.results['Vx'][0]  # Constant axial velocity at design
        
        # Generate flow coefficient range if not provided
        if flow_coefficients is None:
            phi_design = Vx_design / (self.omega * self.rmean)
            flow_coefficients = np.linspace(0.3 * phi_design, 1.2 * phi_design, 15)
        
        # Arrays to store characteristic data (at mean radius)
        m_dot_array = []
        Q_array = []
        Delta_p0_array = []
        Delta_p_static_array = []
        incidence_array = []
        diffusion_factor_array = []
        power_array = []
        
        rho = 1.225  # kg/m³
        A_annulus = np.pi * (self.rcas**2 - self.rhub**2)
        
        # Sweep through flow coefficients
        for phi in flow_coefficients:
            # New axial velocity
            Vx_new = phi * self.omega * self.rmean
            
            # Radial distribution
            r = np.linspace(self.rhub, self.rcas, n_points)
            U = self.omega * r
            U_mean = self.omega * self.rmean
            idx_mean = n_points // 2
            
            # INLET: purely axial
            Vtheta_1 = np.zeros_like(r)
            Vx_array = np.ones_like(r) * Vx_new
            
            # Inlet relative velocity
            Wtheta_1 = Vtheta_1 - U
            W_1 = np.sqrt(Vx_array**2 + Wtheta_1**2)
            Beta_1 = np.degrees(np.arctan(Wtheta_1 / Vx_array))
            
            # OUTLET: Keep RELATIVE outlet angles FIXED at design point (blade geometry)
            # Beta_2 = arctan(Wtheta_2 / Vx) stays constant
            # So: Wtheta_2 = Vx * tan(Beta_2)
            Beta_2 = Beta_2_design.copy()  # Fixed relative outlet angle
            Wtheta_2 = Vx_array * np.tan(np.radians(Beta_2))
            
            # Absolute tangential velocity changes with Vx
            # Wtheta = Vtheta - U, so Vtheta = Wtheta + U
            Vtheta_2 = Wtheta_2 + U
            
            # Outlet relative velocity
            W_2 = np.sqrt(Vx_array**2 + Wtheta_2**2)
            
            # Work (Euler equation)
            Delta_h = U * (Vtheta_2 - Vtheta_1)
            
            # Pressures
            Delta_p0 = rho * Delta_h
            V_2_squared = Vx_array**2 + Vtheta_2**2
            Delta_p_static = Delta_p0 - 0.5 * rho * V_2_squared
            
            # Mass flow
            m_dot = rho * Vx_new * A_annulus
            Q = m_dot / rho  # m³/s
            
            # Incidence angle (change in magnitude of inlet angle from design point)
            # Incidence = |β1| - |β1_design|
            # Positive incidence = steeper angle, higher loading (occurs at low flow)
            # Negative incidence = shallower angle (occurs at high flow)
            Beta_1_design = self.results['Beta_1'][idx_mean]
            incidence = np.abs(Beta_1[idx_mean]) - np.abs(Beta_1_design)
            
            # De Haller and diffusion factor
            DeHaller_off = W_2[idx_mean] / W_1[idx_mean]
            
            # Solidity with stagger
            spacing = 2 * np.pi * r[idx_mean] / self.Nblade
            stagger = 0.5 * (Beta_1[idx_mean] + Beta_2[idx_mean])
            C_true = self.Cx / np.cos(np.radians(stagger))
            solidity = C_true / spacing
            
            Delta_Vtheta = Vtheta_2[idx_mean] - Vtheta_1[idx_mean]
            D = 1 - DeHaller_off + (np.abs(Delta_Vtheta) / (2 * solidity * W_1[idx_mean]))
            
            # Power (shaft power = mass flow * work)
            power = m_dot * Delta_h[idx_mean]
            
            # Store results at mean radius
            m_dot_array.append(m_dot)
            Q_array.append(Q)
            Delta_p0_array.append(Delta_p0[idx_mean])
            Delta_p_static_array.append(Delta_p_static[idx_mean])
            incidence_array.append(incidence)
            diffusion_factor_array.append(D)
            power_array.append(power)
        
        # Store characteristic data
        self.characteristic = {
            'flow_coefficients': flow_coefficients,
            'm_dot': np.array(m_dot_array),
            'Q': np.array(Q_array),
            'Delta_p0': np.array(Delta_p0_array),
            'Delta_p_static': np.array(Delta_p_static_array),
            'incidence': np.array(incidence_array),
            'diffusion_factor': np.array(diffusion_factor_array),
            'power': np.array(power_array)
        }
        
        return self.characteristic
    
    def plot_characteristic(self):
        """
        Plot fan characteristic curves
        """
        if not hasattr(self, 'characteristic'):
            print("Run generate_characteristic() first!")
            return
        
        char = self.characteristic
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Pressure rise vs. flow rate
        axes[0, 0].plot(char['Q']*3600, char['Delta_p0'], 'b-o', linewidth=2, label='Stagnation Δp₀')
        axes[0, 0].plot(char['Q']*3600, char['Delta_p_static'], 'r-s', linewidth=2, label='Static Δp')
        axes[0, 0].set_xlabel('Volume Flow Rate (m³/h)')
        axes[0, 0].set_ylabel('Pressure Rise (Pa)')
        axes[0, 0].set_title('Fan Characteristic Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power vs. flow rate
        axes[0, 1].plot(char['Q']*3600, char['power'], 'g-o', linewidth=2)
        axes[0, 1].set_xlabel('Volume Flow Rate (m³/h)')
        axes[0, 1].set_ylabel('Shaft Power (W)')
        axes[0, 1].set_title('Power Consumption')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Incidence angle vs. flow rate
        axes[1, 0].plot(char['Q']*3600, char['incidence'], 'm-o', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Design point')
        axes[1, 0].set_xlabel('Volume Flow Rate (m³/h)')
        axes[1, 0].set_ylabel('Incidence Angle (deg)')
        axes[1, 0].set_title('Incidence Angle (Off-Design)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diffusion factor vs. flow rate
        axes[1, 1].plot(char['Q']*3600, char['diffusion_factor'], '-o', color='brown', linewidth=2)
        axes[1, 1].axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Max limit (0.6)')
        axes[1, 1].set_xlabel('Volume Flow Rate (m³/h)')
        axes[1, 1].set_ylabel('Diffusion Factor')
        axes[1, 1].set_title('Lieblein Diffusion Factor')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figs/fan_characteristic.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Example usage
    """
    # Define design parameters
    rhub = 0.015       # Hub radius (m) - 15mm
    rcas = 0.040       # Casing radius (m) - 40mm
    Cx = 0.015         # Axial chord (m) - 15mm
    Nblade = 7         # Number of blades
    Beta_1m = 60       # RELATIVE inlet angle at mean (degrees)
    vort_exp = 0.0      # Free vortex (applies to absolute frame)
    rpm = 3000         # Rotational speed
    DeHaller = 0.75    # De Haller number W2/W1 at midspan
    
    # Create fan design object
    fan = AxialFanDesign(rhub, rcas, Cx, Nblade, Beta_1m, vort_exp, rpm, DeHaller)
    
    # Calculate design
    fan.calculate_design(n_points=50)
    
    # Print summary
    fan.print_summary()
    
    # Generate plots
    fan.plot_radial_distributions()
    fan.plot_velocity_triangles(radii_fraction=[0.0, 0.5, 1.0])
    
    # Generate and plot characteristic curves
    print("\nGenerating fan characteristic curves...")
    fan.generate_characteristic()
    fan.plot_characteristic()


if __name__ == "__main__":
    main()
