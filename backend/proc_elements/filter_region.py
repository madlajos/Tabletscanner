def characterize_particles(
    data,
    use_filtered=True,
    selected_columns=None,
    include_excluded=False,
    debug=False
):
    """
    A detect_particles eredményéből táblázat épít.

    use_filtered:
        True  -> csak a szűrt szemcséket írja ki
        False -> az összeset

    selected_columns:
        pl. ["particle_id", "image_index", "label", "area_px", "circularity", "intensity_mean"]
        ha None, akkor az összes mezőt visszaadja
    """

    if data["error"] is not None:
        return data

    if "meta" not in data or "particles" not in data["meta"]:
        data["error"] = "E3100"
        return data

    if "results" not in data or data["results"] is None:
        data["results"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    if selected_columns is not None and not isinstance(selected_columns, (list, tuple)):
        data["error"] = "E3108"
        return data

    particles_source = data["meta"]["particles_filtered"] if use_filtered and "particles_filtered" in data["meta"] else data["meta"]["particles"]

    table = []

    for image_particles in particles_source:
        for particle in image_particles:
            excluded = bool(particle.get("excluded", False))
            if excluded and not include_excluded:
                continue

            if selected_columns is None:
                row = dict(particle)
            else:
                row = {col: particle.get(col, None) for col in selected_columns}

            table.append(row)

    data["results"]["particle_table"] = table
    data["meta"]["particle_table_config"] = {
        "use_filtered": use_filtered,
        "selected_columns": list(selected_columns) if selected_columns is not None else None,
        "include_excluded": include_excluded
    }

    data["history"].append("characterize_particles")

    if debug:
        print("Particle table complete")
        print(f"Rows: {len(table)}")
        if len(table) > 0:
            print(table[0])

    return data
