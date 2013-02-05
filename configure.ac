AC_INIT([RGPULasso],1.0)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])

AC_ARG_WITH([r-include],
	AC_HELP_STRING([--with-r-include=r_include],
		[specify the full path to your R header files, e.g. /usr/share/R/include]),
	[R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
	AC_HELP_STRING([--with-r-lib=r-lib],
		[specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
	[R_LIB="$withval"])

AC_ARG_WITH([rcpp-home],
	AC_HELP_STRING([--with-rcpp-home=rcpp-home],
		[specify the full path of the Rcpp package directory, can get from evaluating "Rcpp:::RcppLdPath()" in R]),
	[RCPP_HOME="$withval"])


withdebug=""
AC_ARG_ENABLE([debug],
	AC_HELP_STRING([--enable-debug],
		[debug the library]),
	[withdebug=$enableval])


device_emu=""
AC_ARG_ENABLE([emulation],
	AC_HELP_STRING([--enable-emulation],
		[emulate the gpu because maybe you do not have one]),
	[device_emu=$enableval])

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
	AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
	CUDA_HOME="/usr/local/cuda"
else
	AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
	AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
	AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
	CUDA_LIB_DIR="/lib"
else
	AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
	CUDA_LIB_DIR="/lib64"
fi

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
	AC_MSG_ERROR("could not determine R_HOME")
else
	if test -z "${R_INCLUDE}"; then
		R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
	else
		R_INCLUDE="-I${R_INCLUDE}"
	fi
	if test -z "${R_LIB}"; then
		R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
	else
		R_LIB="-L${R_LIB} -lR"
	fi
	if test -z "${RCCP_HOME}"; then
		RCPP_HOME=`${R_HOME}/bin/Rscript -e "cat(tools::file_path_as_absolute(base::system.file(package='Rcpp')))"`
		RCPP_LD_PATH=`${R_HOME}/bin/Rscript -e "cat(Rcpp:::RcppLdPath())"`
	fi
	RCPP_LIB="-L${RCPP_HOME} -lRcpp"
	AC_MSG_RESULT("using ${R_INCLUDE} for R header files")
	AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
	AC_MSG_RESULT("using ${RCPP_LIB} for RCpp")
fi

if test "${device_emu}" = yes ; then
	EMULATION="-deviceemu"
else
	EMULATION=""
fi

if test "${withdebug}" = yes ; then
	CUDA_DEBUG="-g -G"
	CPP_DEBUG="-g"
else
	CUDA_DEBUG=""
	CPP_DEBUG=""
fi

AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_LIB_DIR)
AC_SUBST(EMULATION)
AC_SUBST(CUDA_DEBUG)
AC_SUBST(CPP_DEBUG)

AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)
AC_SUBST(RCPP_LIB)
AC_SUBST(RCPP_HOME)
AC_SUBST(RCPP_LD_PATH)


AC_SUBST(DEVICEOPTS)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT




