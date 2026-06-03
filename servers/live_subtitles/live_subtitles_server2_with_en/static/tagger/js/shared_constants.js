// ===== Global State & Constants =====
(function () {
  // tags.html state
  window.charts = {};
  window.activeSegmentFilter = null;
  window.currentSegmentCount = 10;

  // Color palette matching Python COLORS
  window.EVENT_COLORS = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
    "#009E73",
    "#56B4E9",
    "#E69F00",
    "#F0E442",
    "#0072B2",
  ];

  // Thresholds matching Python constants
  window.HIGH_PROBABILITY_THRESHOLD = 0.7;
  window.MEDIUM_PROBABILITY_THRESHOLD = 0.4;
  window.DEFAULT_PROBABILITY_THRESHOLD = 0.3;

  // Dashboard state
  window.selectedFile = null;

  // Speech classes (fetched from config)
  window.SPEECH_CLASS_NAMES = [
    "Speech",
    "Speech synthesizer",
    "Narration, monologue",
    "Conversation",
    "Dialogue",
    "Babbling",
    "Children shouting",
    "Shout",
    "Whispering",
    "Laughter",
    "Crying",
    "Sighing",
    "Singing",
    "Humming",
    "Mantra",
    "Rapping",
    "Yodeling",
    "Chant",
    "Crowd",
    "Cheering",
    "Applause",
  ];
})();
