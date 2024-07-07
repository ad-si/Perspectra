import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import numpy as np
    from skimage.measure import approximate_polygon
    import plotly.express as px
    return approximate_polygon, mo, np, px


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Polygon Simplification

        Given is a list of vertices which describe a polygon. The last edge is implicitly defined by connecting the last vertex to the first vertex of the list.

        For testing, we use following example polygon:
        """
    )
    return


@app.cell
def __(mo, np, px):
    points = np.array(
        [
            [14, 46],
            [14, 140],
            [234, 144],
            [234, 47],
            [44, 30],
        ]
    )
    mo.ui.plotly(
        px.scatter(
            x=points[:, 0],
            y=points[:, 1],
            width=600,
            height=300,
        )
    )
    return points,


@app.cell
def __():
    def vertices_to_edges(vertices):
        edges = []
        for index, coordinate in enumerate(vertices):
            if index != (len(vertices) - 1):
                edges.append([coordinate, vertices[index + 1]])
        return edges
    return vertices_to_edges,


@app.cell
def __(np, vertices_to_edges):
    # TODO: Does not seem to work
    def simplify_polygon(vertices, targetCount=4):
        """
        Merge globally shortest edge with shortest neighbor
        """
        if not np.array_equal(vertices[0], vertices[-1]):
            raise ValueError(
                f"First vertex ({vertices[0]}) \
                and last ({vertices[-1]}) must be the same"
            )

        edges = np.array(vertices_to_edges(vertices))

        while len(edges) > targetCount:
            edges_lengths = [np.linalg.norm(edge[0] - edge[1]) for edge in edges]
            edge_min_index = np.argmin(edges_lengths)
            edge_prev_length = edges_lengths[edge_min_index - 1]
            edge_next_length = np.take(
                edges_lengths,
                edge_min_index + 1,
                mode="wrap",
            )

            if edge_prev_length < edge_next_length:
                # Merge with previous edge
                edges[edge_min_index][0] = edges[edge_min_index - 1][0]
                edges = np.delete(edges, edge_min_index - 1, axis=0)
                edges_lengths = np.delete(edges_lengths, edge_min_index - 1)
            else:
                # Merge with next edge
                edges[edge_min_index][1] = edges[
                    (edge_min_index + 1) % len(edges)
                ][1]
                edges = np.delete(edges, (edge_min_index + 1) % len(edges), axis=0)
                edges_lengths = np.delete(
                    edges_lengths, (edge_min_index + 1) % len(edges)
                )

        # Re-add first vertex to close polygon
        vertices_new = np.append(edges[:, 0], [edges[0][0]], axis=0)
        return vertices_new
    return simplify_polygon,


@app.cell
def __(mo, np, points, px, simplify_polygon):
    # Re-add first point to end of array
    _points_wrapped = np.vstack((points, points[0]))
    _simplified_polygons = simplify_polygon(_points_wrapped)
    mo.ui.plotly(
        px.scatter(
            x=_simplified_polygons[:, 0],
            y=_simplified_polygons[:, 1],
            width=600,
            height=300,
        )
    )
    return


@app.cell
def __(approximate_polygon):
    def reduce_polygon_to_4_points_new(corners_sorted, epsilon=0.1):
        reduced_polygon = corners_sorted
        while len(reduced_polygon) > 4:
            reduced_polygon = approximate_polygon(
                corners_sorted,
                tolerance=epsilon,
            )
            epsilon += 0.1

        return reduced_polygon
    return reduce_polygon_to_4_points_new,


@app.cell
def __(mo, np, points, px):
    # Re-add first point to end of array
    corners_sorted_wrapped = np.vstack((points, points[0]))
    mo.ui.plotly(
        px.scatter(
            x=corners_sorted_wrapped[:, 0],
            y=corners_sorted_wrapped[:, 1],
            width=600,
            height=300,
        )
    )
    return corners_sorted_wrapped,


@app.cell
def __(corners_sorted_wrapped, mo, px, reduce_polygon_to_4_points_new):
    # TODO: Still picks the wrong 4 points
    _corners_reduced = reduce_polygon_to_4_points_new(corners_sorted_wrapped)
    mo.ui.plotly(
        px.scatter(
            x=_corners_reduced[:, 0],
            y=_corners_reduced[:, 1],
            width=600,
            height=300,
        )
    )
    return


@app.cell
def __(np):
    def reduce_polygon(polygon, angle_th=0, distance_th=0):
        angle_th_rad = np.deg2rad(angle_th)
        points_removed = [0]

        while len(points_removed):
            points_removed = list()
            for i in range(0, len(polygon) - 2, 2):
                v01 = polygon[i - 1] - polygon[i]
                v12 = polygon[i] - polygon[i + 1]
                d01 = np.linalg.norm(v01)
                d12 = np.linalg.norm(v12)
                if d01 < distance_th and d12 < distance_th:
                    points_removed.append(i)
                    continue
                angle = np.arccos(np.sum(v01 * v12) / (d01 * d12))
                if angle < angle_th_rad:
                    points_removed.append(i)
            polygon = np.delete(polygon, points_removed, axis=0)

        return polygon
    return reduce_polygon,


@app.cell
def __(mo, points, px, reduce_polygon):
    _reduced_polygons = reduce_polygon(points, angle_th=1, distance_th=5)
    mo.ui.plotly(
        px.scatter(
            x=_reduced_polygons[:, 0],
            y=_reduced_polygons[:, 1],
            width=600,
            height=300,
        )
    )
    return


if __name__ == "__main__":
    app.run()
