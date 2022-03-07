Dynamic Nanobrain Emulator
===========================

Code package to simulate, in time, dynamical networks based on a physical nanowire design.
A proper description of the physical devices and the model that describes their dynamical response will be published elsewhere.

The core project files like ''network.py'', ''time_marching.py'' are located at the top-level repository, while tutorials and larger applications are separated into specific modules.

For a general introduction about the use of the package and how to create networks, see the files in tutorials, like ''Tutorial.py''.

More advanced use case examples are found under applications. 

To generate the documentation, please use 

$ cd docs

$ sphinx-apidoc -o . ../

$ make html

Application to random recurrent networks
----------------------------------------

One of the applications to highlight here is the capability of the nodes to act collectively as a recurrent network.
We have proved that such a network has the so called Echo State Property by training a random network to perform as a frequency generator.
The files describing the application are found in the folder ''echostatenetwork''.

.. image:: https://github.com/DavidWinge/dynamic-nanobrain/blob/d45c986ec0b8f17ab9e39c22310e9f8e7c0ee71f/docs/network_layout_cropped.png

Written 2021 by David Winge
