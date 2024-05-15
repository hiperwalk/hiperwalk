<h1 align="center">
	<picture>
	  <source media="(prefers-color-scheme: dark)"
		  srcset="https://raw.githubusercontent.com/hiperwalk/hiperwalk.github.io/main/static/images/logo_dark.png"
		  width="150">
	  <img src="https://raw.githubusercontent.com/hiperwalk/hiperwalk.github.io/main/static/images/logo.png"
	       width="150">
	</picture>
	<br>
	Hiperwalk
</h1>
<br>

[![PyPI Downloads](https://img.shields.io/pypi/dm/hiperwalk.svg?label=PyPI%20downloads)](
https://pypi.org/project/hiperwalk/)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-Ask%20questions-blue.svg)](
https://stackoverflow.com/questions/tagged/hiperwalk)
[![Quantum Week Paper](https://img.shields.io/badge/DOI-10.1109%2FQCE57702.2023.00055-blue)](
https://doi.org/10.1109/QCE57702.2023.00055)
[![Contribute](https://img.shields.io/badge/Contribute-Good%20First%20Issue-green.svg)](
https://github.com/hiperwalk/hiperwalk/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+First+Issue%22)


Hiperwalk, an acronym for High-Performance Quantum Walk Simulator, is a powerful,
open-source software designed to simulate quantum walks on various graph structures.
Leveraging heterogeneous High-Performance Computing (HPC),

* **Website:** [http://hiperwalk.org/](http://hiperwalk.org/)
* **Documentation:** [https://hiperwalk.org/docs/stable/documentation/](https://hiperwalk.org/docs/stable/documentation/)
* **Source code:** [https://github.com/hiperwalk/hiperwalk](https://github.com/hiperwalk/hiperwalk)
* **Contributing:** [https://hiperwalk.org/docs/stable/development/index.html](https://hiperwalk.org/docs/stable/development/index.html)
* **Bug reports:** [https://github.com/hiperwalk/hiperwalk/issues](https://github.com/hiperwalk/hiperwalk/issues)
* **Tutorial:** [https://hiperwalk.org/docs/stable/tutorial](https://hiperwalk.org/docs/stable/tutorial)
* **Examples:** [https://github.com/hiperwalk/hiperwalk/tree/master/examples](https://github.com/hiperwalk/hiperwalk/tree/master/examples)


# Install

Hiperwalk is available with HPC support or not.

## No HPC Support
Install the lastest version of Hiperwalk from PyPi

```shell
pip install hiperwalk
```

## HPC Support

HPC support is available via Docker or local installation.
The latter requires
[hiperblas-core](https://github.com/hiperblas/hiperblas-core) and
[pyhiperblas](https://github.com/hiperblas/pyhiperblas)
for HPC on CPU, with optional GPU support provided by
[hiperblas-opencl-bridge](https://github.com/hiperblas/hiperblas-opencl-bridge).
For more details, access
[https://hiperwalk.org/docs/stable/install/](https://hiperwalk.org/docs/stable/install/).

# Call for Contributions

Small improvements or fixes are always appreciated.
If you are considering larger contributions to the source code, or
if you are unsure where to start or how your skills fit in,
please contact us through
[hiperwalk@gmail.com](mailto:hiperwalk@gmail.com).

There are multiple ways to contribute to Hiperwalk.
For instance, you can:
- contribute to [HiperBLAS](https://github.com/hiperblas)
- help us stay on top of new and old [issues](https://github.com/hiperwalk/hiperwalk/issues)
- develop tutorials, presentations, and other educational materials
- maintain and improve [our website](https://github.com/hiperwalk/hiperwalk.github.io)
- develop graphic design for our brand assets and promotional materials
- translate website content
- help with outreach and onboard new contributors

If you are new to contributing to open source, [this
guide](https://opensource.guide/how-to-contribute/) helps explain why, what,
and how to successfully get involved.
