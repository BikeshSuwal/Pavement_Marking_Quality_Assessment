    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v11",
        tooltip={
            "text": "Condition: {predicted_condition}\nLat: {latitude}\nLon: {longitude}"
        }
    ))
