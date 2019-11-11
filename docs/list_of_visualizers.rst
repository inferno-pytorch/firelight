List of Visualizers
-------------------

.. currentmodule:: firelight.visualizers.base

The following non-container visualizers are currently available.
They all derive from :class:`BaseVisualizer`.

..
    inheritance-diagram:: firelight.visualizers.visualizers
    :top-classes: firelight.visualizers.base.BaseVisualizer
    :parts: 1

..
    inheritance-diagram:: firelight.visualizers.container_visualizers
    :top-classes: firelight.visualizers.base.ContainerVisualizer
    :parts: 1

.. automodsumm:: firelight.visualizers.visualizers
    :classes-only:
    :skip: PCA, TSNE, BaseVisualizer

.. currentmodule:: firelight.visualizers.base

These are the available visualizers combining multiple visualizations.
Their base class is the :class:`ContainerVisualizer`.

.. automodsumm:: firelight.visualizers.container_visualizers
    :classes-only:
    :skip: ContainerVisualizer
