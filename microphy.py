#!/usr/bin/env python3

import scipy.special
import numpy
import matplotlib.pyplot as plt
import thermo

class MicrophyBase2M(object):
    """
    This class is a base class for representing a 2-moment distribution of a microphysical specie.
    Parameters specific to a given distribution shape are defined in subclasses.
    Whatever is the distribution shape, some specie parameters are needed: a, b, c, d
    """
    rho0 = 101300. / (thermo.Thermo.Rd * (thermo.Thermo.Tt+20))

    def __init__(self, a, b, c, d, use_masked_arrays=True):
        """
        Parameters of the specie must be set
        """
        self.param = dict(a=a, b=b, c=c, d=d)
        self.__dict__.update(self.param)
        self._use_masked_arrays = use_masked_arrays

    def N_by_bin(self, vm, N, bin_edges):
        """
        Given a total volumic mass (vm) and the total number concentration (N), returns the number concentration in each bin
        if bin_edges is [0., 1.e-6, 2.e-6], 2 bins are used: (0., 1.e-6) and (1.e-6, 2.e-6)
        You can use a huge value (e.g. 999) as the last edge to get all the remaining part of the distribution
        """
        if len(bin_edges) < 2:
            raise ValueError("We must have, at least, 2 edges to form a bin")
        return self.N_in_bin(vm, N, bin_edges[:-1], bin_edges[1:])

    def N_in_bin(self, vm, N, Dmin, Dmax):
        """
        given a volumic mass (vm) and the total number concentration (N), returns the number concentration between Dmin and Dmax
        """
        return N * self.partM(vm, N, 0, Dmin, Dmax)


    def vm_by_bin(self, vm, N, bin_edges):
        """
        Given a total volumic mass (vm) and the total number concentration (N), returns the volumic masses in each bin
        if bin_edges is [0., 1.e-6, 2.e-6], 2 bins are used: (0., 1.e-6) and (1.e-6, 2.e-6)
        You can use a huge value (e.g. 999) as the last edge to get all the remaining part of the distribution
        """
        if len(bin_edges) < 2:
            raise ValueError("We must have, at least, 2 edges to form a bin")
        return self.vm_in_bin(vm, N, bin_edges[:-1], bin_edges[1:])

    def vm_in_bin(self, vm, N, Dmin, Dmax):
        """
        given a volumic mass (vm) and the total number concentration (N), returns the volumic mass between Dmin and Dmax
        """
        return self.a * N * self.partM(vm, N, self.b, Dmin, Dmax)

    def find_D(self, vm, N, vmUnder=None, vmAbove=None):
        """
        Given a volumic mass (vm) and the total number concentration (N), finds the diameter for which volumic mass with inferior (resp. superior) diameter is vmUnder (resp. vmAbove).
        Only one of vmUnder and vmAbove must be set
        """
        if vmUnder is None and vmAbove is None:
            raise ValueError("Only one of vmUnder and vmAbove must be set")
        elif vmUnder is not None and vmAbove is not None:
            raise ValueError("One of vmUnder and vmAbove must be set")
        else:
            if not isinstance(vm, numpy.ndarray):
                if isinstance(N, numpy.ndarray):
                    raise ValueError("vm and N must have the same type and shape")
                vm_array = numpy.array([vm])
                N_array = numpy.array([N])
                searchedVM = vmUnder if vmUnder is not None else vm_array - vmAbove
                if isinstance(searchedVM, numpy.ndarray):
                    if searchedVM.shape != vm_array.shape or len(searchedVM) != len(vm_array):
                        raise ValueError("If vmUnder/vmAbove is an array, it must be a 1D array of same length as vm")
                else:
                    searchedVM = numpy.array([searchedVM])
                convert = True
            else:
                if (not isinstance(N, numpy.ndarray)) or N.shape != vm.shape:
                    raise ValueError("vm and N must have the same type and shape")
                N_array = N
                vm_array = vm
                searchedVM = vmUnder if vmUnder is not None else vm_array - vmAbove
                if not isinstance(searchedVM, numpy.ndarray):
                    raise ValueError("If vm is an array, vmUnder/vmAbove must be an array.")
                else:
                    if searchedVM.shape != vm_array.shape or len(searchedVM) != len(vm_array):
                        raise ValueError("If vm is an array, vmUnder/vmAbove must be aa array of same shape and length as vm")
                convert = False
            if self._use_masked_arrays:
                result = numpy.ma.masked_all_like(vm_array)
            else:
                result = numpy.zeros_like(vm_array)
            mask = vm_array>0.
            lambda_specie_mask = self.Lambda(vm_array[mask], N_array[mask])
            result[mask] = self._compute_find_D(searchedVM[mask], N_array[mask], lambda_specie_mask)
            return result[0] if convert else result

    def _compute_find_D(self, searchVM, N, lambda_specie_mask):
        """
        Computational part of find_D which is dependent to the distribution shape
        """
        raise NotImplementedError("This method must be implemented in a distribution class.")

    def terminal_speed_velocity(self, D, rho=rho0):
        """
        Given a diameter, returns the terminal speed velocity
        """
        return self.c * D ** self.d * (self.rho0/rho)**0.4

    def mean_terminal_speed_velocity(self, vm, N, rho=rho0):
        """
        Given a volumic mass (vm) and the total number concentration (N), returns the mean terminal speed velocity
        """
        return self.c * (self.rho0/rho)**0.4 * self.G(self.b+self.d) / self.G(self.b) * self.Lambda(vm, N)**(-self.d)

    def mean_terminal_speed_velocity_in_bin(self, vm, N, Dmin, Dmax, rho=rho0, nullSpeed=numpy.nan):
        """
        Given a volumic mass (vm) and the total number concentration (N), returns the mean terminal speed velocity
        """
        vm_in_bin = self.vm_in_bin(vm, N, Dmin, Dmax)
        result = vm_in_bin.copy()
        result[vm_in_bin == 0.] = nullSpeed
        mask = vm_in_bin != 0.
        result[mask] =  (N * (self.c * (self.rho0 / rho) ** 0.4 * self.a) * \
                         self.partM(vm, N, self.b + self.d, Dmin, Dmax))[mask] / \
                        vm_in_bin[mask]
        return result

    def mean_terminal_speed_velocity_by_bin(self, vm, N, bin_edges, rho=rho0, nullSpeed=numpy.nan):
        """
        Given a total volumic mass (vm) and the total number concentration (N), returns the mean terminal speed velocity in each bin
        if bin_edges is [0., 1.e-6, 2.e-6], 2 bins are used: (0., 1.e-6) and (1.e-6, 2.e-6)
        You can use a huge value (e.g. 999) as the last edge to get all the remaining part of the distribution
        When volumic mass is null, speed is undefined. nullSpeed will be affected to these points
        """
        if len(bin_edges) < 2:
            raise ValueError("We must have, at least, 2 edges to form a bin")
        return self.mean_terminal_speed_velocity_in_bin(vm, N, bin_edges[:-1], bin_edges[1:], rho, nullSpeed)

    def g(self, vm, N, D):
        """
        Normalized distribution
        """
        raise NotImplementedError("This method must be implemented in a distribution class.")
       
    def Lambda(self, vm, N): #With an upper case letter because lambda seems to be an invalid name for a method
        """
        Given a volumic mass (vm) and the total number concentration (N), computes the lambda parameter
        """
        return (vm / (self.a * N * self.G(self.b)))**(1./(-self.b))

    def G(self, p):
        """
        G function wich returns the <p>th moment of the distribution
        """
        raise NotImplementedError("This method must be implemented in a distribution class.")

    def M(self, vm, N, p):
        """
        Given a volumic mass (vm) and the total number concentration (N), returns the p^th moment of the law
        """
        return self.G(p) / self.Lambda(vm, N) ** p

    def partM(self, vm, N, p, Dmin, Dmax):
        """
        Given a volumic mass (vm) and the total number concentration (N), returns the part of M between Dmin and Dmax (use 999 as infinite value) for the <p>th moment
        """
        assert isinstance(vm, numpy.ndarray) == isinstance(N, numpy.ndarray), 'vm and N must be both scalars or arrays'
        if isinstance(Dmin, numpy.ndarray) != isinstance(Dmax, numpy.ndarray):
            raise ValueError("Dmin and Dmax must be both scalar or ndarray alltogether")
        if not isinstance(vm, numpy.ndarray):
            vm_array = numpy.array([vm])
            N_array = numpy.array([N])
            if isinstance(Dmin, numpy.ndarray):
                if len(Dmin.shape) != 1 or len(Dmax.shape) != 1 or Dmin.shape != Dmax.shape:
                    raise ValueError("Dmin and Dmax must be scalars or 1D arrays (of same length) when vm is a scalar")
                Dmin_array = Dmin.reshape(Dmin.shape[0], 1)
                Dmax_array = Dmax.reshape(Dmin.shape[0], 1)
                convert_D = False
            else:
                Dmin_array = numpy.array([Dmin]).reshape((1,1))
                Dmax_array = numpy.array([Dmax]).reshape((1,1))
                convert_D = True
            convert_vm = True
        else:
            if (not isinstance(Dmin, numpy.ndarray)) or Dmin.shape != Dmax.shape:
                raise ValueError("Dmin and Dmax must be arrays (with Dmin.shape==Dmax.shape) when vm is an array")
            if len(vm.shape) != 1:
                raise ValueError("vm must be a scalar or a 1D array")
            if len(Dmin.shape) == 1:
                if Dmin.shape != vm.shape:
                    raise ValueError("If vm is an array and Dmin/Dmax 1D arrays, their shape must be the same as vm")
                Dmin_array=Dmin.reshape(1, Dmin.shape[0])
                Dmax_array=Dmin.reshape(1, Dmin.shape[0])
                convert_D = True
            elif len(Dmin.shape) == 2:
                if Dmin.shape[1] != vm.shape[0]:
                    raise ValueError("If vm is an array and Dmin/Dmax 2D arrays, axis 1 of Dmin/Dmax must be in agrrement with vm size")
                Dmin_array = Dmin
                Dmax_array = Dmax
                convert_D = False
            else:
                raise ValueError("Dmin and Dmax cannot have more than 2 axis")
            vm_array = vm
            N_array = N
            convert_vm = False

        if self._use_masked_arrays:
            result = numpy.ma.masked_all_like(Dmin_array)
        else:
            result = numpy.zeros_like(Dmin_array)
        mask = vm_array > 0.
        lambda_specie_mask = self.Lambda(vm_array[mask], N_array[mask])
        result[:, mask] = self._compute_partM(lambda_specie_mask, p, Dmin_array[:, mask], Dmax_array[:, mask])
        if convert_vm:
            if convert_D:
                result = result[0, 0]
            else:
                result = result[:, 0]
        else:
            if convert_D:
                result = result[0, :]
        return result

    def _compute_partM(self, lambda_specie_mask, p, Dmin_array, Dmax_array):
        """
        Computational part of partM which is dependent to the distribution shape
        """
        raise NotImplementedError("This method must be implemented in a distribution class.")

    def _discretize(self, vm, N, p, mult=1., norm=False, cum=False, edgeRatio=100, num=200):
        """
        Discretize a distribution
        :param vm: volumic mass
        :param N: total number concentration
        :param p: number of the moment
        :param mult: multiplicative factor
        :param norm: if False, curves are also multiplied by the total number
                     not sure of the meaning of norm=True when (p, mult) != (0., 1.)
        :param cum: True to discretize the cumulative distribution
        :param edgeRatio: discretize disgarding vm/edgeRatio on each side of the distribution
        :param num: number of points of the disretization
        :return: the tuple (diameters, values)
        """
        Dmin, Dmax = self.find_D(vm, N, vmUnder=vm / edgeRatio), self.find_D(vm, N, vmAbove=vm / edgeRatio)
        D = numpy.linspace(Dmin, Dmax, num)
        if cum:
            actual_mult = mult
        else:
            actual_mult = mult * D**p
        if not norm:
            actual_mult = actual_mult * N
        if cum:
            return (D, numpy.vectorize(self.partM)(vm, N, p, 0., D) * actual_mult)
        else:
            return (D, numpy.vectorize(self.g)(vm, N, D) * actual_mult)
        

    def plot_distribution(self, vm, N, p, mult=1., title="", ylabel="", norm=False, cum=False, ax=None, colors=None, styles=None):
        """
        Given a volumic mass (vm) and the total number concentration (N), plots the <p>th moment of the distribution
        Curves are multiplied by mult
        if not norm, curves are also multiplied by the total number
           not sure of the meaning of norm=True when (p, mult) != (0., 1.)
        if cum, cumulative distribution is plotted
        colors and styles are list of colors and line styles to use
        """
        if colors is None:
            colors = ['black', 'silver', 'red', 'darksalmon', 'gold', 'olivedrab',
                      'chartreuse', 'skyblue', 'darkblue', 'purple', 'magenta']
        if styles is None:
            styles = ['-', '--', '-.', ':']
        typeY = [(c, ls) for ls in styles for c in colors]
        if ax is None:
            fig = plt.figure()
            use_ax = fig.add_subplot(1, 1, 1)
        else:
            use_ax = ax
        try:
            vm[0]
            vm_list = vm
            N_list = N
        except:
            vm_list = [vm]
            N_list = [N]
        result = []
        for iValue, value in enumerate(vm_list):
            D, dist = self._discretize(value, N_list[iValue], p, mult, norm, cum)
            line, = use_ax.plot(D * 1000., dist, label="vm=" + str(value), color=typeY[iValue][0], linestyle=typeY[iValue][1])
            result.append(line)
        if len(vm_list) > 1:
            use_ax.legend()
        use_ax.set_xlabel("Diameter (mm)")
        use_ax.set_ylabel(ylabel)
        use_ax.set_title(title + ((" with vm=" + str(vm) + " and N=" + str(N)) if len(vm_list) == 1 else " with different vm and/or N"))
        if ax is None:
            plt.show()
        return result

    def plot_number_distribution(self, vm, N, specie_name, norm=False, cum=False, ax=None, colors=None, styles=None):
        """
        Given a volumic mass (vm) and the total number concentration (N), plots the number distribution
        specie_name is the name of the specie to use on plot
        if cum, cumulative distribution is plotted
        colors and styles are list of colors and line styles to use
        """
        if cum:
            title = "Cumulative number distribution for " + specie_name
            ylabel = "Number concentration (#/m^3)"
        else:
            title = "Number distribution for " + specie_name
            ylabel = "Number concentration\ndensity (#/m per m^3)"
        return self.plot_distribution(vm, N, 0, title=title, ylabel=ylabel, norm=norm, cum=cum, ax=ax, colors=colors, styles=styles)

    def plot_mass_distribution(self, vm, N, specie_name, cum=False, ax=None, colors=None, styles=None):
        """
        Given a volumic mass (vm) and the total number concentration (N), plots the mass distribution
        specie_name is the name of the specie to use on plot
        if cum, cumulative distribution is plotted
        colors and styles are list of colors and line styles to use
        """
        if cum:
            title = "Cumulative mass distribution for " + specie_name
            ylabel = "Volumic mass (kg/m^3)"
        else:
            title = "Mass distribution for " + specie_name
            ylabel = "Volumic mass density\n(kg/m per m^3)"
        return self.plot_distribution(vm, N, self.b, mult=self.a, title=title, ylabel=ylabel, cum=cum, ax=ax, colors=colors, styles=styles)

    def plot_mass_number(self, vm, N, specie_name, ax=None, colors=None, styles=None):
        """
        Given a volumic mass (vm) and the total number concentration (N), plots the mass - number relation
        specie_name is the name of the specie to use on plot
        colors and styles are list of colors and line styles to use
        """
        if colors is None:
            colors = ['black', 'silver', 'red', 'darksalmon', 'gold', 'olivedrab',
                      'chartreuse', 'skyblue', 'darkblue', 'purple', 'magenta']
        if styles is None:
            styles = ['-', '--', '-.', ':']
        typeY = [(c, ls) for ls in styles for c in colors]
        if ax is None:
            fig = plt.figure()
            use_ax = fig.add_subplot(1, 1, 1)
        else:
            use_ax = ax
        try:
            vm[0]
            vm_list = vm
            N_list = N
        except:
            vm_list = [vm]
            N_list = [N]
        result = []
        for iValue, value in enumerate(vm_list):
            D, dist_N = self._discretize(value, N_list[iValue], 0)
            D, dist_mass = self._discretize(value, N_list[iValue], self.b, self.a)
            line, = use_ax.plot(dist_N, dist_mass, label="vm=" + str(value), color=typeY[iValue][0], linestyle=typeY[iValue][1])
            result.append(line)
        if len(vm_list) > 1:
            use_ax.legend()
        use_ax.set_xlabel("Number concentration density (#/m per m^3)")
        use_ax.set_ylabel("Volumic mass density\n(kg/m per m^3)")
        title = "Mass/number relation"
        use_ax.set_title(title + ((" with vm=" + str(vm) + " and N=" + str(N)) if len(vm_list) == 1 else " with different vm and/or N"))
        if ax is None:
            plt.show()
        return result

class MicrophyBase1M(object):
    """
    This class contain the non distribution specific code part that allows to transform a 2-moment object into a one-moment object
    To enable the computation of a total numer concentration for a one-moment specie, some extra parameters are needed: C and x
    """

    def __init__(self, C, x):
        """
        Parameters needed to compute total number concentration must be set
        """
        self.param = dict(C=C, x=x)

    def _add_methods(self):
        """
        This method defines new methods with the N argument suppressed
        This needs a 2-moment underlying object (self._m2)
        """
        self.param.update(self._m2.param)
        self.__dict__.update(self.param)
        for symb in ['N_by_bin',
                     'N_in_bin',
                     'vm_by_bin',
                     'vm_in_bin',
                     'find_D',
                     'mean_terminal_speed_velocity',
                     'mean_terminal_speed_velocity_in_bin',
                     'mean_terminal_speed_velocity_by_bin',
                     'g',
                     'M',
                     'partM',
                     'plot_distribution',
                     'plot_number_distribution',
                     'plot_mass_distribution',
                     'plot_mass_number']:
            def fct(vm, *args2, _symb=symb, **kwargs2):
                N = self.number_concentration(vm)
                return getattr(self._m2, _symb)(vm, N, *args2, **kwargs2)
            self.__dict__[symb] = fct
        self.G = self._m2.G

    def number_concentration(self, vm):
        """
        Given a volumic mass (vm), returns the total number concentration
        """
        if not isinstance(vm, numpy.ndarray):
            return self.number_concentration(numpy.array([vm]))[0]
        else:        
            result = vm.copy()
            result[vm == 0.] = 0.
            result[vm != 0.] = self.C * self.Lambda(vm[vm != 0.]) ** self.x
            return result

    def Lambda(self, vm):  # With an upper case letter because lambda seems to be an invalid name for a method
        """
        Given a volumic mass (vm), computes the lambda parameter
        """
        return (vm / (self.a * self.C * self.G(self.b))) ** (1. / (self.x - self.b))

class MicrophyGeneralizedGamma2M(MicrophyBase2M):
    """
    This class represent a generalized gamma distribution of a microphysical specie.
    Distribution is specified with parameters alpha and nu
    g(D) = alpha/Gamma(nu) * lambda**(alpha*nu) * D**(alpha*nu-1) * exp(-(lambda*D)**alpha)
    """

    def __init__(self, alpha, nu, a, b, c, d, use_masked_arrays=True):
        """
        Parameters of the distribution must be set
        """
        super().__init__(a, b, c, d, use_masked_arrays=use_masked_arrays)
        self.param.update(dict(alpha=alpha, nu=nu))
        self.__dict__.update(self.param)

    def _compute_find_D(self, searchVM, N, lambda_specie_mask):
        """
        Computational part of find_D which is dependent to the distribution shape
        """
        result = searchVM * lambda_specie_mask ** self.b / (self.a * N * self.G(self.b))
        result = scipy.special.gammaincinv(self.b/self.alpha+self.nu, result)**(1./self.alpha)/lambda_specie_mask

        return result

    def g(self, vm, N, D):
        """
        Normalized distribution
        """
        return self.alpha / scipy.special.gamma(self.nu) * self.Lambda(vm, N)**(self.alpha*self.nu)*D**(self.alpha*self.nu-1)*numpy.exp(-(self.Lambda(vm, N)*D)**self.alpha)
       
    def G(self, p):
        """
        G function wich returns the <p>th moment of the distribution
        """
        return scipy.special.gamma(self.nu+p/self.alpha)/scipy.special.gamma(self.nu)

    def _compute_partM(self, lambda_specie_mask, p, Dmin_array, Dmax_array):
        """
        Computational part of partM which is dependent to the distribution shape
        """
        result = lambda_specie_mask ** (-p) * self.G(p) * (
                   scipy.special.gammainc(p/self.alpha+self.nu, (lambda_specie_mask * Dmax_array)**self.alpha) -
                   scipy.special.gammainc(p/self.alpha+self.nu, (lambda_specie_mask * Dmin_array)**self.alpha)
                 )
        return result

class MicrophyGeneralizedGamma1M(MicrophyBase1M):
    def __init__(self, alpha, nu, a, b, c, d, C, x, use_masked_arrays=True):
        super().__init__(C=C, x=x)
        self._m2 = MicrophyGeneralizedGamma2M(alpha, nu, a, b, c, d, use_masked_arrays=use_masked_arrays)
        self._add_methods()

class MicrophyLognormal2M(MicrophyBase2M):
    """
    This class represent a lognormal distribution of a microphysical specie.
    Distribution is specified with parameter sigma
    g(D) = 1/(sqrt(2*pi)*sigma*D) * exp(-(log(lambda*D) / (sigma*sqrt(2)))**2) according to Meso-NH ICE3/4 doc
    """

    def __init__(self, sigma, a, b, c, d, use_masked_arrays=True):
        """
        Parameters of the distribution must be set
        """
        super().__init__(a, b, c, d, use_masked_arrays=use_masked_arrays)
        self.param.update(dict(sigma=sigma))
        self.__dict__.update(self.param)

    def _compute_find_D(self, searchVM, N, lambda_specie_mask):
        """
        Computational part of find_D which is dependent to the distribution shape
        """
        result = 2 * searchVM * lambda_specie_mask ** self.b / (self.a * N * self.G(self.b)) -1
        result = (scipy.special.erfinv(result) + self.b * self.sigma / numpy.sqrt(2.)) * self.sigma * numpy.sqrt(2.)
        result = numpy.exp(result) / lambda_specie_mask

        return result

    def g(self, vm, N, D):
        """
        Normalized distribution
        """
        return 1. / (numpy.sqrt(2. * numpy.pi) * self.sigma * D) * numpy.exp(-(numpy.log(self.Lambda(vm, N)*D)/(numpy.sqrt(2.) * self.sigma))**2)
       
    def G(self, p):
        """
        G function wich returns the <p>th moment of the distribution
        """
        return numpy.exp(p**2 * self.sigma**2 / 2.)

    def _compute_partM(self, lambda_specie_mask, p, Dmin_array, Dmax_array):
        """
        Computational part of partM which is dependent to the distribution shape
        """
        lower_part = numpy.where(Dmin_array == 0., -1.,
                                 scipy.special.erf(numpy.log(lambda_specie_mask * Dmin_array)/(self.sigma * numpy.sqrt(2.)) - p*self.sigma/numpy.sqrt(2)))
        result = lambda_specie_mask ** (-p) * self.G(p) * 0.5 * (
                   scipy.special.erf(numpy.log(lambda_specie_mask * Dmax_array)/(self.sigma * numpy.sqrt(2.)) - p*self.sigma/numpy.sqrt(2)) -
                   lower_part
                 )
        return result

class MicrophyLognormal1M(MicrophyBase1M):
    def __init__(self, sigma, a, b, c, d, C, x, use_masked_arrays=True):
        super().__init__(C=C, x=x)
        self._m2 = MicrophyLognormal2M(sigma, a, b, c, d, use_masked_arrays=use_masked_arrays)
        self._add_methods()

def microphyFactory(specie, use_masked_arrays=True):
    """
    This function returns the instance corresponding to the specie
    """
    #ICE3 and ICE4 species
    if specie == 'ICE:c_land':
        return MicrophyGeneralizedGamma2M(alpha=1, nu=3, a=524., b=3., c=3.2e7, d=2., use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:c_sea':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=1, a=524., b=3., c=3.2e7, d=2., use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:r':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=1, a=524., b=3., c=842., d=.8, C=8.e6, x=-1., use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:i':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=3, a=0.82, b=2.5, c=800., d=1., use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:s':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=1, a=.02, b=1.9, c=5.1, d=.27, C=5., x=1., use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:g':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=1, a=19.6, b=2.8, c=124., d=.66, C=5.e5, x=-.5, use_masked_arrays=use_masked_arrays)
    elif specie == 'ICE:h':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=8, a=470., b=3., c=207., d=.64, C=4.e4, x=-1., use_masked_arrays=use_masked_arrays)

    #LIMA species
    elif specie == 'LIMA:c':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=1, a=1000.*numpy.pi/6, b=3., c=2.98e7, d=2., use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:r':
        return MicrophyGeneralizedGamma2M(alpha=1, nu=2, a=1000.*numpy.pi/6, b=3., c=842., d=.8, use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:i_plates':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=3, a=0.82, b=2.5, c=747., d=1., use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:i_columns':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=3, a=2.14e-3, b=1.7, c=1.96e5, d=1.585, use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:i_rosettes':
        return MicrophyGeneralizedGamma2M(alpha=3, nu=3, a=44., b=3., c=4.e5, d=1.663, use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:s':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=1, a=.02, b=1.9, c=5., d=.27, C=5., x=1., use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:g':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=1, a=19.6, b=2.8, c=122., d=.66, C=5.e5, x=-.5, use_masked_arrays=use_masked_arrays)
    elif specie == 'LIMA:h':
        return MicrophyGeneralizedGamma1M(alpha=1, nu=8, a=470., b=3., c=201., d=.64, C=4.e4, x=-1., use_masked_arrays=use_masked_arrays)

    #
    elif specie == 'testLogN1:r':
        return MicrophyLognormal1M(sigma=.6, a=524., b=3., c=842., d=.8, C=8.e6, x=-1., use_masked_arrays=use_masked_arrays)
    elif specie == 'testLogN2:r':
        return MicrophyLognormal2M(sigma=.6, a=524., b=3., c=842., d=.8, use_masked_arrays=use_masked_arrays)
    else:
        raise ValueError('Unknonw specie')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes some distribution related quantities and plot distributions. If specie is a one-moment specie, you must not give a value for N, otherwise (2-moment specie), you must specify the N value.')
    parser.add_argument('-s', '--specie', dest='specie', action='store', required=True, help="specie to use (e.g.: 'ICE:c_land', 'ICE:c_sea', 'ICE:r', ..., 'LIMA:c', ..., 'LIMA:i_plates', ...)")
    parser.add_argument('-v', '--volumicMass', dest='vm', type=float, action='store', required=True, help='volumic mass to use (rho times mixing ratio: kg/m^3)')
    parser.add_argument('-N', '--number', dest='N', type=float, action='store', default=None, help='total number concentration (#/m^3)')
    parser.add_argument('-p', '--plot', dest='doPlot', action='store_const', const=True, default=False, help='to plotnumber and mass (density -by default- or cumulative) distributions')
    parser.add_argument('-c', '--cum', dest='plotCum', action='store_const', const=True, default=False, help='plots cumulative distributions instead of density distributions')
    parser.add_argument('-m', '--multiple', dest='plotMult', action='store_const', const=True, default=False, help='plots curves for (volumic mass), (volumic mass)*2 and (volumic mass)/2 instead of only (volumic mass)')
    args = parser.parse_args()

    specie = args.specie
    vm = args.vm
    N = args.N

    m = microphyFactory(specie)
    if isinstance(m, MicrophyBase1M) and N is not None:
        raise AttributeError("You must NOT give a value to N if specie is a one-moment specie.")
    elif (not isinstance(m, MicrophyBase1M)) and N is None:
        raise AttributeError("You MUST give a value to N if specie is a two-moment specie.")
    print("Parameters for " + specie + ": " + str(m.param))
    print("Provided volumic mass is: " + str(vm))
    if N is None:
        print("Total number concentration will be diagnosed from the volumic mass value.")
    else:
        print("Provided number concentration is: " + str(N))
    bin_number = 10
    if N is None:
        Lambda = m.Lambda(vm)
        Dmin100, Dmax100 = m.find_D(vm, vmUnder=vm/100.), m.find_D(vm, vmAbove=vm/100.)
        bin_edges = numpy.append(numpy.insert(numpy.linspace(Dmin100, Dmax100, bin_number - 1), 0, 0.), 999.)
        vm_by_bin = m.vm_by_bin(vm, bin_edges)
        mean_terminal_speed_velocity_by_bin = m.mean_terminal_speed_velocity_by_bin(vm, bin_edges)
        N_by_bin = m.N_by_bin(vm, bin_edges)
        mean_terminal_speed_velocity = m.mean_terminal_speed_velocity(vm)
    else:
        Lambda = m.Lambda(vm, N)
        Dmin100, Dmax100 = m.find_D(vm, N, vmUnder=vm/100.), m.find_D(vm, N, vmAbove=vm/100.)
        bin_edges = numpy.append(numpy.insert(numpy.linspace(Dmin100, Dmax100, bin_number - 1), 0, 0.), 999.)
        vm_by_bin = m.vm_by_bin(vm, N, bin_edges)
        mean_terminal_speed_velocity_by_bin = m.mean_terminal_speed_velocity_by_bin(vm, N, bin_edges)
        N_by_bin = m.N_by_bin(vm, N, bin_edges)
        mean_terminal_speed_velocity = m.mean_terminal_speed_velocity(vm, N)
    print("Lambda: " + str(Lambda))
    print("Diameter under which only 1 percent of volumic mass lies: " + str(Dmin100*1.e6) + " micrometers")
    print("Diameter above which only 1 percent of volumic mass lies: " + str(Dmax100*1.e6) + " micrometers")
    for Dmin, Dmax in [(0., 1.), (0., 0.01), (0., 0.001), (0.001, 0.01), (Dmin100, Dmax100)]:
        if N is None:
            vm_in_bin = m.vm_in_bin(vm, Dmin, Dmax)
        else:
            vm_in_bin = m.vm_in_bin(vm, N, Dmin, Dmax)
        print("Volumic mass between diameters " + str(Dmin * 1.e6) + " micrometers and " + str(Dmax * 1.e6) + " micrometers: " + str(vm_in_bin))
    print("Mean terminal speed velocity: " + str(mean_terminal_speed_velocity))
    print("Dmin, Dmax, vm, N, speed by bin for " + str(bin_number) + " bins:")
    acc_speed = 0.
    acc_vm = 0.
    acc_N = 0.
    for i, bin_vm in enumerate(vm_by_bin):
        bin_speed = mean_terminal_speed_velocity_by_bin[i]
        bin_N = N_by_bin[i]
        acc_vm += bin_vm
        acc_speed += bin_speed * bin_vm
        acc_N += bin_N
        print("   " + str(bin_edges[i]) + " " + str(bin_edges[i+1]) + " " + str(bin_vm) + " " + str(bin_N) + " " + str(bin_speed))
    print("Volumic mass computed from bins: " + str(acc_vm))
    print("Mean terminal speed velocity computed from bins: " + str(acc_speed/acc_vm))
    print("Total number concentration computed from bins: " + str(acc_N))
    print("Concentration number (#/m^3): " + str(m.number_concentration(vm) if N is None else N))
    if args.doPlot:
        fig, axes = plt.subplots(nrows=3, ncols=1)
        if N is None:
            if args.plotMult:
                m.plot_number_distribution([vm/2, vm, vm*2], specie, cum=args.plotCum, ax=axes[0])
                m.plot_mass_distribution([vm/2, vm, vm*2], specie, cum=args.plotCum, ax=axes[1])
                m.plot_mass_number([vm/2, vm, vm*2], specie, ax=axes[2])
            else:
                m.plot_number_distribution(vm, specie, cum=args.plotCum, ax=axes[0])
                m.plot_mass_distribution(vm, specie, cum=args.plotCum, ax=axes[1])
                m.plot_mass_number(vm, specie, ax=axes[2])
        else:
            if args.plotMult:
                m.plot_number_distribution([vm/2, vm, vm*2], [N, N, N], specie, cum=args.plotCum, ax=axes[0])
                m.plot_mass_distribution([vm/2, vm, vm*2], [N, N, N], specie, cum=args.plotCum, ax=axes[1])
                m.plot_mass_number([vm/2, vm, vm*2], [N, N, N], specie, ax=axes[2])
            else:
                m.plot_number_distribution(vm, N, specie, cum=args.plotCum, ax=axes[0])
                m.plot_mass_distribution(vm, N, specie, cum=args.plotCum, ax=axes[1])
                m.plot_mass_number(vm, N, specie, ax=axes[2])
        fig.tight_layout()
        plt.show()
