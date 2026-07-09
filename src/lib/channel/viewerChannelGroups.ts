import type { Channel, ChannelGroup } from "@/lib/stores/documentStore";
import { findSourceChannel } from "@/lib/stores/documentStore";

/** One channel row in the Viv / export view of a document channel group. */
export type ViewerChannel = {
  name: string;
  color: string;
  contrast: [number, number];
};

/** Document channel group mapped for Viv settings and exhibit export. */
export type ViewerChannelGroup = {
  State: { Expanded: boolean };
  channels: ViewerChannel[];
  name: string;
  g: number;
};

/** @deprecated Use {@link ViewerChannelGroup}. */
export type ExhibitChannelGroupView = ViewerChannelGroup;

export function buildViewerChannelGroups(
  channelGroups: ChannelGroup[],
  sourceChannels: Channel[],
): ViewerChannelGroup[] {
  return channelGroups.map((group, g) => {
    const { name, channels: groupChannelsList, expanded } = group;
    const channels = groupChannelsList.map((group_channel) => {
      const defaults = { name: "" };
      const { r, g: gg, b } = group_channel.color;
      const color = ((1 << 24) + (r << 16) + (gg << 8) + b)
        .toString(16)
        .slice(1);
      const { lowerLimit, upperLimit } = group_channel;
      const flat = findSourceChannel(sourceChannels, group_channel.channelId);
      const { name: chName } = flat || defaults;
      return {
        color,
        name: chName,
        contrast: [lowerLimit, upperLimit] as [number, number],
      };
    });
    return {
      State: { Expanded: expanded ?? false },
      g,
      name,
      channels,
    };
  });
}
