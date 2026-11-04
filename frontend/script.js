// DWLR Groundwater Evaluation
// Dashboard functionality

const data = window.DWLR_DATA;

// Check that data.js has loaded
if (!data) {
    console.error("DWLR_DATA not found. Check that data.js loads before script.js.");
}

// -----------------------------
// CURRENT DATE
// -----------------------------

const dateElement = document.getElementById("currentDate");

if (dateElement) {
    dateElement.textContent = new Date().toLocaleDateString("en-IN", {
        day: "numeric",
        month: "long",
        year: "numeric"
    });
}

// -----------------------------
// KPI CARDS
// -----------------------------

function updateKpis() {
    if (!data) return;

    const values = {
        totalLocations: data.kpis.locations,
        totalObservations: data.kpis.observations,
        increasingTrends: data.kpis.increasingTrends,
        decreasingTrends: data.kpis.decreasingTrends
    };

    Object.entries(values).forEach(([id, value]) => {
        const element = document.getElementById(id);

        if (element) {
            element.textContent = Number(value).toLocaleString("en-IN");
        }
    });
}

// -----------------------------
// GROUNDWATER CHART
// -----------------------------

function createGroundwaterChart() {
    const canvas = document.getElementById("groundwaterChart");

    if (!canvas || !data || typeof Chart === "undefined") return;

    new Chart(canvas, {
        type: "line",

        data: {
            labels: data.chart.labels,

            datasets: [{
                label: "Groundwater Level (mbgl)",
                data: data.chart.values,
                borderColor: "#0d9488",
                backgroundColor: "rgba(13, 148, 136, 0.12)",
                borderWidth: 3,
                fill: true,
                tension: 0.35,
                pointRadius: 4,
                pointBackgroundColor: "#0d9488"
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: true
                }
            },

            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: "Water Level (mbgl)"
                    }
                },

                x: {
                    title: {
                        display: true,
                        text: "Month"
                    }
                }
            }
        }
    });
}

// -----------------------------
// GROUNDWATER MAP
// -----------------------------

function createGroundwaterMap() {
    const mapElement = document.getElementById("groundwaterMap");

    if (!mapElement || !data || typeof L === "undefined") return;

    const map = L.map("groundwaterMap").setView([22.5, 80], 5);

    L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: "Tiles &copy; Esri",
            maxZoom: 19
        }
    ).addTo(map);

    data.mapMarkers.forEach(marker => {
        L.marker([marker.latitude, marker.longitude])
            .addTo(map)
            .bindPopup(`<b>${marker.name}</b><br>Illustrative map marker`);
    });
}

// -----------------------------
// TREND TABLE
// -----------------------------

function populateTrendTable() {
    const tableBody = document.getElementById("trendTableBody");

    if (!tableBody || !data) return;

    tableBody.innerHTML = "";

    data.trendRows.forEach(row => {
        const tableRow = document.createElement("tr");

        [
            row.location,
            row.district,
            row.state,
            row.trend,
            row.status
        ].forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value;
            tableRow.appendChild(cell);
        });

        tableBody.appendChild(tableRow);
    });
}

// -----------------------------
// NAVIGATION
// -----------------------------

function setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");

    navItems.forEach(item => {
        item.addEventListener("click", event => {
            const pageName = item.textContent.trim();

            if (pageName.includes("Historical Analysis")) {
                event.preventDefault();
                window.location.href = "historical.html";
            }
        });
    });
}

// -----------------------------
// INITIALIZE DASHBOARD
// -----------------------------

document.addEventListener("DOMContentLoaded", () => {
    updateKpis();
    createGroundwaterChart();
    createGroundwaterMap();
    populateTrendTable();
    setupNavigation();
});