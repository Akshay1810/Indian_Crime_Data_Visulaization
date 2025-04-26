// assets/clientside.js
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = {
  // Function to toggle the drawer and overlay
  toggleDrawer: function (n_open, n_close, n_overlay, currentDrawerClass) {
    // Determine which button triggered the callback
    const ctx = dash_clientside.callback_context;
    const triggeredId = ctx.triggered_id;

    // console.log("Current Class:", currentDrawerClass, "Trigger:", triggeredId);

    let drawerClass = currentDrawerClass || "drawer"; // Default class
    let overlayClass = "drawer-overlay"; // Default overlay class

    if (
      triggeredId === "btn-open-drawer" &&
      drawerClass.indexOf("open") === -1
    ) {
      // console.log("Opening drawer");
      drawerClass = "drawer open";
      overlayClass = "drawer-overlay visible";
      // Optional: Add class to body to prevent scrolling or push content
      // document.body.classList.add("drawer-open");
    } else if (
      (triggeredId === "btn-close-drawer" ||
        triggeredId === "drawer-overlay") &&
      drawerClass.indexOf("open") !== -1
    ) {
      // console.log("Closing drawer");
      drawerClass = "drawer"; // Remove open class
      overlayClass = "drawer-overlay"; // Remove visible class
      // Optional: Remove class from body
      // document.body.classList.remove("drawer-open");
    } else {
      // If triggered unexpectedly or state is already correct, don't change
      // console.log("No change needed");
      return [dash_clientside.no_update, dash_clientside.no_update];
    }

    // console.log("New Classes:", drawerClass, overlayClass);
    return [drawerClass, overlayClass];
  },

  // Function to handle clicks on drawer links
  handleDrawerLinkClick: function (...args) {
    const ctx = dash_clientside.callback_context;
    const triggeredInput = ctx.triggered[0];
    const linkId = triggeredInput.prop_id.split(".")[0];

    if (!linkId || !linkId.startsWith("link-")) {
      return [dash_clientside.no_update, "drawer", "drawer-overlay"]; // No update if trigger is unexpected
    }

    // Extract tab value from link ID (e.g., 'link-tab-statewise' -> 'tab-statewise')
    const targetTabValue = linkId.substring("link-".length);

    // Close drawer and overlay, update tab value
    // console.log("Switching to tab:", targetTabValue);
    // Optional: Remove class from body
    // document.body.classList.remove("drawer-open");
    return [targetTabValue, "drawer", "drawer-overlay"];
  },

  // Function to update the active link style
  updateActiveLink: function (activeTabValue, ...linkIds) {
    // linkIds is an array containing the actual IDs passed via State
    // console.log("Updating active link style. Active tab:", activeTabValue);
    // console.log("Link IDs:", linkIds);

    const outputs = [];
    linkIds.forEach((linkId) => {
      // Extract the tab value this link corresponds to
      const linkTabValue = linkId.substring("link-".length);
      let className = "drawer-link"; // Default class
      if (linkTabValue === activeTabValue) {
        className += " active"; // Add active class if it matches
      }
      outputs.push(className);
    });

    // console.log("Output classes:", outputs);
    // Return the array of class names, one for each link Output
    return outputs;
  },
};
