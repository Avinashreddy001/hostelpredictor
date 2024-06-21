// Populate location dropdown dynamically (example using JavaScript)
const locationSelect = document.getElementById("location");
const locations = [/* Your location data */];

locations.forEach(location => {
  const option = document.createElement("option");
  option.value = location;
  option.text = location;
  locationSelect.appendChild(option);
});
