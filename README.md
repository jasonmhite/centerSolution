# Coordinate conventions

Grid is `width` columns by `height` rows. Coordinates are given in the order `(row, col)` to follow Numpy/C array
convention, and start from 0. The origin is the upper left coordinates following Matplotlib convention.

When dimensions of arrays are given in the description they are assumed to be height then width. This is mostly for
sanity purposes, you can transpose them and things should work as long as you also permute coordinates.