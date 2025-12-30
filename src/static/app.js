document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    document.getElementById('refresh-btn').addEventListener('click', fetchData);

    // Zoom Buttons
    document.querySelectorAll('.zoom-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Remove active class from all
            document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
            // Add to clicked
            e.target.classList.add('active');

            const range = e.target.getAttribute('data-range');
            zoomChart(range);
        });
    });
});

let globalDates = [];
let globalLow = [];
let globalHigh = [];

async function fetchData() {
    const btn = document.getElementById('refresh-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '↻ Loading...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/data');
        const data = await response.json();

        if (data.error) {
            console.error('API Error:', data.error);
            return;
        }

        globalDates = data.dates;
        globalLow = data.low;
        globalHigh = data.high;
        renderChart(data);
        updateHeader(data);

        // Apply active zoom
        const activeBtn = document.querySelector('.zoom-btn.active');
        if (activeBtn) {
            zoomChart(activeBtn.getAttribute('data-range'));
        }
    } catch (error) {
        console.error('Fetch Error:', error);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function updateHeader(data) {
    // Update Price
    const priceEl = document.getElementById('current-price');
    priceEl.textContent = data.last_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    // Colorize price
    priceEl.className = 'info-value up'; // Default
    if (data.close.length > 1 && data.last_price < data.close[data.close.length - 2]) {
        priceEl.className = 'info-value down';
    }

    // Update Signal
    const signalEl = document.getElementById('current-signal');
    signalEl.textContent = data.current_signal;
    signalEl.className = `signal-badge signal-${data.current_signal}`;

    // Info
    const scoreEl = document.getElementById('trend-score');
    scoreEl.textContent = data.current_score;
    // Colorize score
    if (data.current_score > 0) scoreEl.className = 'info-value up';
    else if (data.current_score < 0) scoreEl.className = 'info-value down';
    else scoreEl.className = 'info-value';

    document.getElementById('last-updated').textContent = data.last_date;
}

function zoomChart(range) {
    if (!globalDates.length) return;

    let startDate;
    const lastDate = new Date(globalDates[globalDates.length - 1]);

    if (range === 'ALL') {
        const update = {
            'xaxis.autorange': true
        };
        Plotly.relayout('main-chart', update);
        return;
    }

    // Calculate start date
    const targetDate = new Date(lastDate);

    switch (range) {
        case '1M': targetDate.setMonth(lastDate.getMonth() - 1); break;
        case '3M': targetDate.setMonth(lastDate.getMonth() - 3); break;
        case '6M': targetDate.setMonth(lastDate.getMonth() - 6); break;
        case '1Y': targetDate.setFullYear(lastDate.getFullYear() - 1); break;
        case '5Y': targetDate.setFullYear(lastDate.getFullYear() - 5); break;
    }

    // Format to YYYY-MM-DD
    const startStr = targetDate.toISOString().split('T')[0];

    // Add 5% padding to the right
    const timeDiff = lastDate.getTime() - targetDate.getTime();
    const paddingMs = timeDiff * 0.05;
    const extendedEndDate = new Date(lastDate.getTime() + paddingMs);
    const endStr = extendedEndDate.toISOString().split('T')[0];

    // Auto-scale Y Axis
    let startIndex = globalDates.findIndex(d => d >= startStr);
    if (startIndex === -1) startIndex = 0;

    // Slice data for visible range
    const visibleLows = globalLow.slice(startIndex);
    const visibleHighs = globalHigh.slice(startIndex);

    if (visibleLows.length > 0) {
        const minPrice = Math.min(...visibleLows);
        const maxPrice = Math.max(...visibleHighs);
        const padding = (maxPrice - minPrice) * 0.05; // 5% padding

        Plotly.relayout('main-chart', {
            'xaxis.range': [startStr, endStr],
            'xaxis.autorange': false,
            'yaxis.range': [minPrice - padding, maxPrice + padding],
            'yaxis.autorange': false
        });
    } else {
        Plotly.relayout('main-chart', {
            'xaxis.range': [startStr, endStr],
            'xaxis.autorange': false
        });
    }
}

function renderChart(data) {
    const dates = data.dates;

    // TradingView Colors
    const colorUp = '#089981';
    const colorDown = '#f23645';
    const gridColor = '#2a2e39';

    // 1. Candlestick
    const candlestick = {
        x: dates,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        type: 'candlestick',
        name: 'ETH-USD',
        increasing: { line: { color: colorUp }, fillcolor: colorUp },
        decreasing: { line: { color: colorDown }, fillcolor: colorDown },
        yaxis: 'y'
    };

    // 2. Trend Score Bars
    const markerColors = data.trend_score.map(score => {
        if (score > 0) return colorUp;
        if (score < 0) return colorDown;
        return '#787b86';
    });

    const trendBars = {
        x: dates,
        y: data.trend_score.map(s => Math.abs(s)),
        type: 'bar',
        name: 'Trend Score',
        marker: {
            color: markerColors
        },
        yaxis: 'y2'
    };

    // 3. Position Strip (Bottom)
    const positionStrip = {
        x: dates,
        y: data.signal.map(s => 1),
        type: 'bar',
        name: 'Position',
        marker: {
            color: data.signal.map(s => s === 1 ? colorUp : '#363c4e') // Green or Dark (Flat)
        },
        yaxis: 'y3',
        hoverinfo: 'none'
    };

    // 4. Entry Markers (Bull Start)
    const entryMarkers = {
        x: data.events.entries.dates,
        y: data.events.entries.prices,
        mode: 'markers+text',
        type: 'scatter',
        name: 'Bull Start',
        marker: {
            symbol: 'triangle-up',
            size: 12,
            color: '#2962ff',
            line: { color: 'white', width: 1 }
        },
        text: data.events.entries.prices.map(p => `Bull Start<br>$${p.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`),
        textposition: 'bottom center',
        textfont: {
            family: 'sans-serif',
            size: 10,
            color: '#2962ff'
        },
        yaxis: 'y'
    };

    // 5. Exit Markers (Trend Break)
    const exitMarkers = {
        x: data.events.exits.dates,
        y: data.events.exits.prices,
        mode: 'markers+text',
        type: 'scatter',
        name: 'Trend Break',
        marker: {
            symbol: 'triangle-down',
            size: 12,
            color: '#be0a2e', // Darker Red for exit
            line: { color: 'white', width: 1 }
        },
        text: data.events.exits.prices.map(p => `Trend Break<br>$${p.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`),
        textposition: 'top center',
        textfont: {
            family: 'sans-serif',
            size: 10,
            color: '#be0a2e'
        },
        yaxis: 'y'
    };

    const layout = {
        plot_bgcolor: '#131722',
        paper_bgcolor: '#131722',
        font: { family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11, color: '#d1d4dc' },
        showlegend: false,
        dragmode: 'pan',

        // Grid Layout
        grid: {
            rows: 3,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom'
        },

        // Axis configuration
        xaxis: {
            rangeslider: { visible: false },
            type: 'date',
            gridcolor: gridColor,
            zeroline: false,
            anchor: 'y3'
        },
        yaxis: {
            domain: [0.35, 1],
            gridcolor: gridColor,
            zeroline: false,
            fixedrange: false,
            side: 'right'
        },
        yaxis2: {
            domain: [0.10, 0.32],
            gridcolor: gridColor,
            zerolinecolor: '#787b86',
            zerolinewidth: 1,
            side: 'right'
        },
        yaxis3: {
            domain: [0.0, 0.05],
            showgrid: false,
            showticklabels: false,
            side: 'right'
        },

        margin: { l: 10, r: 60, t: 10, b: 20 },
        height: window.innerHeight - 55
    };

    const config = {
        responsive: true,
        scrollZoom: true,
        displayModeBar: false // Cleaner look
    };

    Plotly.newPlot('main-chart', [candlestick, trendBars, positionStrip, entryMarkers, exitMarkers], layout, config);

    // Handle Window Resize
    window.onresize = function () {
        Plotly.relayout('main-chart', {
            width: document.querySelector('.chart-area').clientWidth,
            height: document.querySelector('.chart-area').clientHeight
        });
    };
}
