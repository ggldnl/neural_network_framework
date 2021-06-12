# Neural Network Framework

Simple framework to create, train and use DFFs

Keep in mind that this is a personal project and that I have no experience with neural networks whatsoever. It served me to learn something about how neural networks works under the hood.

The classes are well commented and there are several examples in the `main.test` package, so I don't think it is necessary to explain how to use of the code even more. Check out the two examples `TestMNIST` and `TestEMNIST`. 

Note that I used the builder pattern to efficiently create the network instead of a simple constructor; the rest (regarding dataset and layer objects) is very straightforward.

To work, the project requires the [gson library](https://github.com/google/gson).