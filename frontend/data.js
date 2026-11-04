// DWLR Groundwater Evaluation
// Central data file for the dashboard

window.DWLR_DATA = {

    // Project-level figures from the current analysis
    kpis: {
        locations: 17235,
        observations: 405078,
        increasingTrends: 2316,
        decreasingTrends: 2117,
        uncertainTrends: 12802
    },

    // Demonstration chart data
    // Replace with actual groundwater observations when connected.
    chart: {
        labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        values: [12.4, 12.7, 12.5, 13.1, 13.4, 13.2]
    },

    // Illustrative map markers only — not actual monitoring wells.
    mapMarkers: [
        {
            name: "Delhi",
            latitude: 28.6139,
            longitude: 77.2090
        },
        {
            name: "Mumbai",
            latitude: 19.0760,
            longitude: 72.8777
        },
        {
            name: "Bengaluru",
            latitude: 12.9716,
            longitude: 77.5946
        },
        {
            name: "Chennai",
            latitude: 13.0827,
            longitude: 80.2707
        },
        {
            name: "Hyderabad",
            latitude: 17.3850,
            longitude: 78.4867
        }
    ],

    // Illustrative trend table rows
    trendRows: [
        {
            location: "Sample Location 1",
            district: "Sample District",
            state: "Sample State",
            trend: "Increasing depth",
            status: "Provisional"
        },
        {
            location: "Sample Location 2",
            district: "Sample District",
            state: "Sample State",
            trend: "Decreasing depth",
            status: "Provisional"
        },
        {
            location: "Sample Location 3",
            district: "Sample District",
            state: "Sample State",
            trend: "Uncertain",
            status: "Provisional"
        }
    ]

};